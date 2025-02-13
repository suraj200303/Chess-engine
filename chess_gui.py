import pygame
import chess
from engine import ChessEngine
# from engine_final import TrainedChessEngine

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 500, 500
SQ_SIZE = WIDTH // 8
PIECE_SCALE = 0.8

# Colors
WHITE = (240, 217, 181)
BLACK = (181, 136, 99)
HIGHLIGHT = (205, 210, 106)

class ChessGUI:
    def __init__(self, engine):
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Chess AI")
        self.engine = engine
        self.flipped = False
        self.selected_square = None
        self.highlighted = []
        self.pieces = self.load_pieces()

    def load_pieces(self):
        pieces = {}
        for color in ['w', 'b']:
            for piece in ['p', 'n', 'b', 'r', 'q', 'k']:
                key = f"{color}{piece}"
                img = pygame.image.load(f'pieces/{key}.png')
                pieces[key] = pygame.transform.smoothscale(
                    img, (int(SQ_SIZE*PIECE_SCALE), int(SQ_SIZE*PIECE_SCALE)))
        return pieces

    def square_to_pos(self, square):
        """Convert internal square index to screen coordinates"""
        if self.flipped:
            row = 7 - (square // 8)
            col = 7 - (square % 8)
        else:
            row = square // 8
            col = square % 8
        return (col * SQ_SIZE, row * SQ_SIZE)

    def pos_to_square(self, pos):
        """Convert screen coordinates to internal square index"""
        x, y = pos
        col = x // SQ_SIZE
        row = y // SQ_SIZE
        if self.flipped:
            col = 7 - col
            row = 7 - row
        return row * 8 + col

    def draw_board(self):
        for row in range(8):
            for col in range(8):
                color = WHITE if (row+col) % 2 == 0 else BLACK
                pygame.draw.rect(self.screen, color,
                               (col*SQ_SIZE, row*SQ_SIZE, SQ_SIZE, SQ_SIZE))

    def draw_pieces(self):
        board = self.engine.board
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                x, y = self.square_to_pos(square)
                piece_key = f"{'w' if piece.color else 'b'}{piece.symbol().lower()}"
                self.screen.blit(self.pieces[piece_key], (x + SQ_SIZE*(1-PIECE_SCALE)/2, 
                                                         y + SQ_SIZE*(1-PIECE_SCALE)/2))

    def highlight_square(self, square, color):
        x, y = self.square_to_pos(square)
        surf = pygame.Surface((SQ_SIZE, SQ_SIZE))
        surf.set_alpha(100)
        surf.fill(color)
        self.screen.blit(surf, (x, y))

    def run(self):
        clock = pygame.time.Clock()
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                if event.type == pygame.MOUSEBUTTONDOWN:
                    square = self.pos_to_square(pygame.mouse.get_pos())
                    piece = self.engine.board.piece_at(square)
                    
                    if self.selected_square is None:
                        if piece and piece.color == self.engine.board.turn:
                            self.selected_square = square
                            self.highlighted = [move.to_square 
                                              for move in self.engine.board.legal_moves
                                              if move.from_square == square]
                    else:
                        move = chess.Move(self.selected_square, square)
                        if move in self.engine.board.legal_moves:
                            self.engine.board.push(move)
                        self.selected_square = None
                        self.highlighted = []

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_f:
                        self.flipped = not self.flipped

            # Draw board state
            self.screen.fill((0, 0, 0))
            self.draw_board()
            
            # Highlight selected square and moves
            if self.selected_square is not None:
                self.highlight_square(self.selected_square, (255, 255, 0))
                for square in self.highlighted:
                    self.highlight_square(square, HIGHLIGHT)
            
            self.draw_pieces()
            
            # AI move
            if not self.engine.board.is_game_over() and self.engine.board.turn == chess.BLACK:
                move = self.engine.get_best_move()
                if move:
                    self.engine.board.push(move)

            pygame.display.flip()
            clock.tick(30)

        pygame.quit()

if __name__ == "__main__":
    engine = ChessEngine()
    gui = ChessGUI(engine)
    gui.run()
