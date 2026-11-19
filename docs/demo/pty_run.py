"""
Run a command inside a pseudo-terminal of an explicit size, so recorders see a real terminal of that size
even when this script itself is not attached to one.

    python docs/demo/pty_run.py <cols> <rows> -- <command> [args...]
"""
import fcntl
import os
import pty
import select
import struct
import subprocess
import sys
import termios


def main():
    cols, rows = int(sys.argv[1]), int(sys.argv[2])
    argv = sys.argv[sys.argv.index("--") + 1:]
    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))

    def attach():  # make the slave the child's controlling terminal
        os.setsid()
        fcntl.ioctl(0, termios.TIOCSCTTY, 0)

    env = {**os.environ, "COLUMNS": str(cols), "LINES": str(rows), "TERM": os.environ.get("TERM", "xterm-256color")}
    proc = subprocess.Popen(argv, stdin=slave, stdout=slave, stderr=slave, preexec_fn=attach, env=env, close_fds=True)
    os.close(slave)
    while True:
        r, _, _ = select.select([master], [], [], 0.2)
        if r:
            try:
                data = os.read(master, 65536)
            except OSError:
                break
            if not data:
                break
            os.write(1, data)
        elif proc.poll() is not None:
            break
    sys.exit(proc.wait())


if __name__ == "__main__":
    main()
