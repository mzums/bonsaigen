import os
import pty
import pyte
from pathlib import Path
import signal
import fcntl
import termios
import struct
import select

COLS = 48
ROWS = 28
MAX_FRAMES = 300

os.makedirs("../dataset/raw", exist_ok=True)

for tree in range(1000):
    screen = pyte.Screen(COLS, ROWS)
    stream = pyte.Stream(screen)

    pid, fd = pty.fork()

    if pid == 0:
        os.environ["TERM"] = "xterm-256color"
        os.execlp("cbonsai", "cbonsai", "-l")

    fcntl.ioctl(
        fd,
        termios.TIOCSWINSZ,
        struct.pack("HHHH", ROWS, COLS, 0, 0),
    )

    frame = 0
    previous = None

    while True:
        ready, _, _ = select.select([fd], [], [], 1.0)

        if not ready:
            break

        try:
            data = os.read(fd, 4096)
        except OSError:
            break

        if not data:
            break

        stream.feed(data.decode("utf-8", errors="ignore"))
        current = "\n".join(screen.display[:-4])

        if current != previous:
            folder = Path(f"../dataset/raw/tree_{tree:04d}")
            folder.mkdir(parents=True, exist_ok=True)

            with open(folder / f"frame_{frame:04d}.txt", "w") as f:
                f.write(current)

            previous = current
            frame += 1

            if frame >= MAX_FRAMES:
                os.kill(pid, signal.SIGTERM)
                break

    os.close(fd)
    try:
        os.waitpid(pid, 0)
    except ChildProcessError:
        pass

print("Done.")