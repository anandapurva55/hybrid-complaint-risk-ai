# Helpers sourced by the session scripts. A session is recorded with asciinema, so what ends up in the
# recording is a REAL terminal: each command is typed with a human-like delay (the typing is scripted)
# and then actually executed, so its output is whatever the command prints at that moment.

DIM=$'\033[2m'; GREEN=$'\033[32m'; RESET=$'\033[0m'

comment() { printf '%s%s%s\n' "$DIM" "$1" "$RESET"; sleep 0.6; }

# type_cmd "<command>": show the prompt, type the command, run it, then pause so the output can be read.
type_cmd() {
  local cmd="$1" pause="${2:-3}"
  printf '%s$%s ' "$GREEN" "$RESET"
  local i
  for ((i = 0; i < ${#cmd}; i++)); do printf '%s' "${cmd:i:1}"; sleep 0.025; done
  sleep 0.4; printf '\n'
  eval "$cmd"
  sleep "$pause"
}
