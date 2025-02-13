import chess
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, losses, optimizers
import os
import gc

# Configuration
BOARD_SIZE = 8
CHANNELS = 20  # Expanded channel features
ACTION_PLANES = 73  # AlphaZero-style move encoding
BATCH_SIZE = 32
EPOCHS = 15
TFRECORD_PATH = "chess_games.tfrecord"

# Enable GPU optimizations
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
tf.keras.mixed_precision.set_global_policy('mixed_float16')

def board_to_tensor(board):
    """Enhanced board representation with game state features"""
    tensor = np.zeros((BOARD_SIZE, BOARD_SIZE, CHANNELS), dtype=np.float32)
    
    # Piece channels (0-11: white/black pieces)
    piece_types = ['P', 'N', 'B', 'R', 'Q', 'K']
    for square in chess.SQUARES:
        row = 7 - (square // 8)
        col = square % 8
        piece = board.piece_at(square)
        if piece:
            channel = piece_types.index(piece.symbol().upper()) + (6 if piece.color else 0)
            tensor[row, col, channel] = 1
    
    # Game state features (12-19)
    tensor[:, :, 12] = board.turn * 1.0  # Turn
    tensor[:, :, 13] = board.fullmove_number / 100.0  # Game progress
    tensor[:, :, 14] = board.halfmove_clock / 50.0  # Capture urgency
    tensor[:, :, 15] = int(board.has_kingside_castling_rights(chess.WHITE))
    tensor[:, :, 16] = int(board.has_queenside_castling_rights(chess.WHITE))
    tensor[:, :, 17] = int(board.has_kingside_castling_rights(chess.BLACK)) 
    tensor[:, :, 18] = int(board.has_queenside_castling_rights(chess.BLACK))
    tensor[:, :, 19] = 1 if board.is_check() else 0  # Check state
    
    return tensor

def move_to_plane(move, board):
    """Convert move to AlphaZero-style 8x8x73 action planes"""
    plane = np.zeros((BOARD_SIZE, BOARD_SIZE, ACTION_PLANES), dtype=np.float32)
    from_row = 7 - (move.from_square // 8)
    from_col = move.from_square % 8
    
    # Queen moves (56 planes)
    directions = [(1,0), (-1,0), (0,1), (0,-1),
                 (1,1), (1,-1), (-1,1), (-1,-1)]
    for i, (dr, dc) in enumerate(directions):
        if (move.to_square - move.from_square) in chess.SquareSet.ray(move.from_square, move.from_square + dr*8 + dc):
            distance = max(abs((move.to_square//8 - move.from_square//8)),
                           abs((move.to_square%8 - move.from_square%8)))
            plane[from_row, from_col, i*7 + (distance-1)] = 1
            break
    
    # Knight moves (8 planes)
    knight_moves = [(2,1), (2,-1), (-2,1), (-2,-1),
                   (1,2), (1,-2), (-1,2), (-1,-2)]
    if (move.to_square - move.from_square) in chess.SquareSet(chess.BB_KNIGHT_ATTACKS[move.from_square]):
        idx = next(i for i, (dr, dc) in enumerate(knight_moves) 
                  if move.to_square == move.from_square + dr*8 + dc)
        plane[from_row, from_col, 56 + idx] = 1
    
    # Underpromotions (9 planes)
    if move.promotion and move.promotion != chess.QUEEN:
        promo_type = [chess.KNIGHT, chess.BISHOP, chess.ROOK].index(move.promotion)
        plane[from_row, from_col, 64 + promo_type*3 + (move.to_square%8 - move.from_square%8 + 1)] = 1
        
    return plane.reshape((8*8*ACTION_PLANES,))

def create_resnet():
    """AlphaZero-style residual network with mixed precision"""
    inputs = layers.Input(shape=(BOARD_SIZE, BOARD_SIZE, CHANNELS), dtype=tf.float32)
    
    # Initial convolution
    x = layers.Conv2D(256, (3,3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Residual blocks
    for _ in range(6):
        residual = x
        x = layers.Conv2D(256, (3,3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(256, (3,3), padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([x, residual])
        x = layers.ReLU()(x)
    
    # Policy head
    policy = layers.Conv2D(ACTION_PLANES, (1,1))(x)
    policy = layers.Flatten(name='policy')(policy)
    
    # Value head
    value = layers.Conv2D(1, (1,1))(x)
    value = layers.Flatten()(value)
    value = layers.Dense(256, activation='relu')(value)
    value = layers.Dense(1, activation='tanh', name='value')(value)
    
    return models.Model(inputs, [policy, value])

class ChessEngine:
    def __init__(self, model=None):
        os.makedirs('checkpoints', exist_ok=True)
        self.model = model or self._build_compiled_model()
        
    def _build_compiled_model(self):
        model = create_resnet()
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss={
                'policy': losses.CategoricalCrossentropy(from_logits=True),
                'value': losses.MeanSquaredError()
            },
            metrics={'policy': 'accuracy', 'value': 'mae'}
        )
        return model

    def _parse_tfrecord(self, example_proto):
        """Parse TFRecord with fixed shapes"""
        feature_description = {
            'board': tf.io.FixedLenFeature([], tf.string),
            'policy': tf.io.FixedLenFeature([], tf.string),
            'value': tf.io.FixedLenFeature([], tf.float32),
        }
        parsed = tf.io.parse_single_example(example_proto, feature_description)
        
        board = tf.io.parse_tensor(parsed['board'], out_type=tf.float32)
        board.set_shape([BOARD_SIZE, BOARD_SIZE, CHANNELS])
        
        policy = tf.io.parse_tensor(parsed['policy'], out_type=tf.float32)
        policy.set_shape([BOARD_SIZE * BOARD_SIZE * ACTION_PLANES])
        
        return board, (policy, parsed['value'])

    def load_dataset(self, filenames):
        """Optimized data pipeline"""
        dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=tf.data.AUTOTUNE)
        dataset = dataset.map(self._parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.shuffle(10000)
        dataset = dataset.batch(BATCH_SIZE)
        return dataset.prefetch(tf.data.AUTOTUNE)

    def train(self, tfrecord_path=TFRECORD_PATH):
        """Memory-efficient training with callbacks"""
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                'checkpoints/epoch_{epoch}.weights.h5',
                save_weights_only=True,
                save_freq='epoch'
            ),
            tf.keras.callbacks.TensorBoard(log_dir='./logs'),
            tf.keras.callbacks.TerminateOnNaN()
        ]
        
        dataset = self.load_dataset([tfrecord_path])
        steps_per_epoch = tf.data.experimental.cardinality(dataset).numpy()
        
        history = self.model.fit(
            dataset,
            epochs=EPOCHS,
            callbacks=callbacks,
            steps_per_epoch=steps_per_epoch
        )
        
        self.model.save('trained_engine.keras')
        return history

def create_tfrecords(games_path='games.csv'):
    """Convert CSV to optimized TFRecord format"""
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    
    with tf.io.TFRecordWriter(TFRECORD_PATH) as writer:
        for chunk in pd.read_csv(games_path, chunksize=1000):
            for _, row in chunk.iterrows():
                try:
                    board = chess.Board()
                    moves = row['moves'].split()
                    result = 1 if row['winner'] == 'white' else -1
                    
                    for move_san in moves:
                        move = board.parse_san(move_san)
                        
                        # Serialize tensors with proper shapes
                        board_tensor = tf.io.serialize_tensor(
                            tf.convert_to_tensor(board_to_tensor(board))
                        )
                        policy = tf.io.serialize_tensor(
                            tf.convert_to_tensor(move_to_plane(move, board))
                        )
                        
                        example = tf.train.Example(features=tf.train.Features(feature={
                            'board': _bytes_feature(board_tensor.numpy()),
                            'policy': _bytes_feature(policy.numpy()),
                            'value': tf.train.Feature(float_list=tf.train.FloatList(value=[result]))
                        }))
                        
                        writer.write(example.SerializeToString())
                        board.push(move)
                        result *= -1  # Alternate perspective
                except Exception as e:
                    print(f"Skipping invalid game: {e}")
                finally:
                    gc.collect()

if __name__ == "__main__":
    create_tfrecords('games.csv')  # First convert CSV to TFRecords
    engine = ChessEngine()
    history = engine.train()
