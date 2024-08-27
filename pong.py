# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "cvxpy",
#     "numpy",
#     "pygame",
#     "mido",
# ]
# ///

import argparse
import bisect
import json
import math
import random
import time
from dataclasses import dataclass
from typing import List

import cvxpy as cp
import numpy as np
import pygame
import mido


def optimize_pong(note_times, screen_width=800, max_dx=1600):
    note_times = np.array(note_times)
    durations = np.diff(note_times)
    n_shots = len(durations)

    # Variables
    paddle_pos = cp.Variable(n_shots + 1)  # even indices are left, odd are right
    ball_abs_dx = cp.Variable(n_shots)  # absolute horizontal velocity

    # Constraints
    constraints = [
        # Paddle must stay on correct half of the screen
        paddle_pos >= 0,
        paddle_pos <= screen_width / 2,
        # Horizontal velocity must be smaller than constant speed
        ball_abs_dx <= max_dx - 1e-4,  # eps to avoid numerical issues
        # Ball must travel the correct distance
        paddle_pos[:-1] + paddle_pos[1:] == cp.multiply(durations, ball_abs_dx),
    ]

    objective = cp.Maximize(cp.sum(paddle_pos))

    # Solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve()

    if prob.status in ["infeasible", "unbounded"]:
        raise cp.SolverError(f"Problem is {prob.status}")

    assert ball_abs_dx.value.max() <= max_dx, ball_abs_dx.value.max()

    return paddle_pos.value, ball_abs_dx.value


# Constants
WINDOW_SIZE = (1280, 720)
CENTER_X, CENTER_Y = WINDOW_SIZE[0] // 2, WINDOW_SIZE[1] // 2
FPS = 60
PADDLE_WIDTH, PADDLE_HEIGHT = 10, 70
BALL_RADIUS = 5
BALL_SPEED = 700
START_WAIT_S = 1.5
END_PADDING_S = 2  # Define end padding constant

# Colors
BLACK = pygame.Color(0, 0, 0)
WHITE = pygame.Color(255, 255, 255)


def is_y_flipped(y: float) -> int:
    """To simulate bounces of the ball off the top and bottom of the screen."""
    return math.floor(y / WINDOW_SIZE[1]) % 2


def rescale_y(y: float) -> float:
    """Rescale the y-coordinate to fit the window, accounting for bounces."""
    if is_y_flipped(y) == 0:
        return y % WINDOW_SIZE[1]
    else:
        return WINDOW_SIZE[1] - (y % WINDOW_SIZE[1])


class Ball:
    """Represents the ball in the game."""

    def __init__(self, x: float, y: float, radius: int):
        self.x, self.y, self.radius = x, y, radius

    def draw(self, screen: pygame.Surface):
        """Draw the ball on the screen."""
        pygame.draw.circle(
            screen, WHITE, (int(self.x), int(rescale_y(self.y))), self.radius
        )


class Paddle:
    """Represents a paddle in the game."""

    def __init__(self, x: int, y: int, width: int, height: int):
        self.rect = pygame.Rect(x, y, width, height)

    def move_to(self, target_x: float, target_y: float):
        """Move the paddle to a target position."""
        self.rect.center = (target_x, rescale_y(target_y))

    def draw(self, screen: pygame.Surface):
        """Draw the paddle on the screen."""
        pygame.draw.rect(screen, WHITE, self.rect)

    @staticmethod
    def vy_to_y(vy: float) -> float:
        """Get the distance from the center of the paddle to hit the ball with the given vertical velocity.
        Top hits straight up, bottom hits straight down, interpolate for other cases."""
        vy /= BALL_SPEED  # normalize vy to [-1, 1]
        return -vy * PADDLE_HEIGHT / 2


class AudioHandler:
    def __init__(self, audio_file: str):
        self.audio_file = audio_file
        self.is_midi = audio_file.lower().endswith(
            ".mid"
        ) or audio_file.lower().endswith(".midi")
        if self.is_midi:
            self.midi_data = mido.MidiFile(audio_file)

    def get_notes(self):
        if not self.is_midi:
            raise ValueError("Cannot extract notes from non-MIDI file")
        notes = []
        current_time = 0
        for msg in self.midi_data:
            current_time += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                notes.append(current_time)
        return sorted(set(notes))

    def play_music(self, delay: float = 0):
        pygame.mixer.music.load(self.audio_file)
        pygame.time.set_timer(pygame.USEREVENT, int(delay * 1000), 1)


@dataclass
class Keyframe:
    t: float
    x: float
    y: float


class KeyframeList:
    def __init__(self, frames: list[Keyframe]):
        self.frames = frames

    def get_position(self, t: float) -> tuple[float, float]:
        """Get the position at time t."""
        left = bisect.bisect_left(self.frames, t, key=lambda x: x.t)

        if left >= len(self.frames):
            return self.frames[-1].x, self.frames[-1].y
        if left == 0:
            return self.frames[0].x, self.frames[0].y

        frame1, frame2 = self.frames[left - 1], self.frames[left]

        # Linear interpolation
        t_diff = frame2.t - frame1.t
        t_ratio = (t - frame1.t) / t_diff if t_diff != 0 else 0

        x = frame1.x + (frame2.x - frame1.x) * t_ratio
        y = frame1.y + (frame2.y - frame1.y) * t_ratio

        return x, y


class PongGame:
    def __init__(
        self, notes: List[float], start_wait_s: float = 3, end_padding_s: float = 2
    ):
        self.notes = notes

        self.paddle_positions, self.ball_speeds = optimize_pong(
            self.notes,
            WINDOW_SIZE[0],
            BALL_SPEED - 50,  # limit horizontal speed to ensure vertical movement
        )

        self.ball = Ball(CENTER_X, CENTER_Y, BALL_RADIUS)
        self.paddles = (
            Paddle(0, 0, PADDLE_WIDTH, PADDLE_HEIGHT),
            Paddle(0, 0, PADDLE_WIDTH, PADDLE_HEIGHT),
        )

        self.start_time = time.time()
        self.score = 0
        self.cur_note_index = 0
        self.start_wait_s = start_wait_s
        self.end_padding_s = end_padding_s
        self.game_end_time = notes[-1] + start_wait_s + end_padding_s  # Add end padding

        (
            self.ball_frames,
            self.left_paddle_frames,
            self.right_paddle_frames,
        ) = self.compute_keyframes(
            self.paddle_positions, self.ball_speeds, start_wait_s
        )

        self.current_time = 0  # Add this line

    def compute_keyframes(
        self, paddle_positions: List[float], ball_dx: List[float], start_wait_s=3
    ):
        ball_frames = []
        left_paddle_frames = []
        right_paddle_frames = []

        # start
        # x positions relative to left edge, y positions relative to top of screen
        ball_frames.append(Keyframe(0, 0, CENTER_Y))
        left_paddle_frames.append(Keyframe(0, 0, CENTER_Y))
        left_paddle_frames.append(Keyframe(1, 0, CENTER_Y))
        right_paddle_frames.append(Keyframe(0, WINDOW_SIZE[0], CENTER_Y))
        right_paddle_frames.append(Keyframe(1, WINDOW_SIZE[0], CENTER_Y))

        ball_y = CENTER_Y
        perceived_vy = 0
        for i, t in enumerate(self.notes):
            is_left_hit = i % 2 == 0
            real_x = CENTER_X
            real_x += -paddle_positions[i] if is_left_hit else paddle_positions[i]

            if i < len(self.notes) - 1:
                vx = ball_dx[i]
                vy = (BALL_SPEED**2 - vx**2) ** 0.5
                vy *= random.choice([-1, 1])
                perceived_vy = -vy if is_y_flipped(ball_y) == 1 else vy

            ball_frames.append(Keyframe(t + start_wait_s, real_x, ball_y))
            paddle_y = rescale_y(ball_y) + Paddle.vy_to_y(perceived_vy)
            if is_left_hit:
                left_paddle_frames.append(Keyframe(t + start_wait_s, real_x, paddle_y))
            else:
                right_paddle_frames.append(Keyframe(t + start_wait_s, real_x, paddle_y))

            if i < len(self.notes) - 1:
                # compute next ball y
                dt = self.notes[i + 1] - self.notes[i]
                ball_y += vy * dt

        return (
            KeyframeList(ball_frames),
            KeyframeList(left_paddle_frames),
            KeyframeList(right_paddle_frames),
        )

    def update(self, dt: float):
        self.current_time += dt  # Update the current time

        # Update ball position
        ball_x, ball_y = self.ball_frames.get_position(self.current_time)
        self.ball.x, self.ball.y = ball_x, ball_y

        # Update paddle positions
        left_x, left_y = self.left_paddle_frames.get_position(self.current_time)
        right_x, right_y = self.right_paddle_frames.get_position(self.current_time)
        self.paddles[0].move_to(left_x, left_y)
        self.paddles[1].move_to(right_x, right_y)

        # Update score
        while (
            self.cur_note_index < len(self.notes)
            and self.current_time >= self.notes[self.cur_note_index] + self.start_wait_s
        ):
            self.score += 1
            self.cur_note_index += 1

    def draw(self, screen: pygame.Surface):
        """Draw all game elements on the screen."""
        screen.fill(BLACK)
        if self.current_time < self.game_end_time:
            self.ball.draw(screen)
        for paddle in self.paddles:
            paddle.draw(screen)
        font = pygame.font.Font("fonts/PressStart2P-Regular.ttf", 32)
        score_text = font.render(f"{self.score}", True, WHITE)
        score_rect = score_text.get_rect(center=(WINDOW_SIZE[0] // 2, 30))
        screen.blit(score_text, score_rect)

    def is_game_over(self):
        return self.current_time > self.game_end_time


def main(audio_file: str, times_file: str = None):
    pygame.init()
    pygame.mixer.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("Pong")

    audio_handler = AudioHandler(audio_file)

    if times_file:
        with open(times_file, "r") as f:
            notes = json.load(f)
    elif audio_handler.is_midi:
        notes = audio_handler.get_notes()
    else:
        raise ValueError("For non-MIDI audio files, you must provide a times_file")

    game = PongGame(notes, start_wait_s=START_WAIT_S, end_padding_s=END_PADDING_S)
    clock = pygame.time.Clock()

    audio_handler.play_music(START_WAIT_S)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.USEREVENT:
                pygame.mixer.music.play()

        # Check if the game is over
        if game.is_game_over():
            running = False
            continue

        dt = clock.tick(FPS) / 1000.0

        game.update(dt)
        game.draw(screen)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pong")
    parser.add_argument(
        "audio_file",
        help="Path to the audio file (MIDI or other audio format)",
    )
    parser.add_argument(
        "--times_file",
        help="Path to a JSON file containing a list of times (required for non-MIDI audio files)",
        default=None,
    )
    args = parser.parse_args()

    if not args.times_file and not (
        args.audio_file.lower().endswith(".mid")
        or args.audio_file.lower().endswith(".midi")
    ):
        parser.error("For non-MIDI audio files, you must provide a times_file")

    main(args.audio_file, times_file=args.times_file)
