# plainsh for bash: type ":" then plain English.
#
#   : delete log files older than 30 days      the line changes colour as soon as you type ":", and the command
#                                              it would become is shown underneath while you type
#   Enter      puts the command on your prompt to read, edit and run (nothing is run for you)
#   Tab        next alternative reading        Up/Down   earlier requests of this terminal
#   Esc, Ctrl-C, or Backspace over the ":"     leave without doing anything
#
# Each terminal keeps one throwaway session file (commands run, requests, corrections), so a request can
# build on the one before it: ": find pdfs over 10mb", then ": delete them".
#
# Settings (set before sourcing):  PLAINSH_COLOR  colour of the request line, as SGR numbers (default 38;5;141)
#                                  PLAINSH_PREVIEW=0  no live preview while typing

[[ $- == *i* && -n ${BASH_VERSION-} ]] || return 0
(( BASH_VERSINFO[0] > 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] >= 4) )) || return 0
__plainsh_bin=${PLAINSH_BIN:-plainsh}
command -v "$__plainsh_bin" >/dev/null 2>&1 || return 0

if [[ -n ${XDG_RUNTIME_DIR-} && -d $XDG_RUNTIME_DIR ]]; then __plainsh_dir=$XDG_RUNTIME_DIR/plainsh; else __plainsh_dir=/tmp/plainsh-$UID; fi
[[ -d $__plainsh_dir ]] || mkdir -p -m 700 "$__plainsh_dir" 2>/dev/null || return 0
__plainsh_session=$__plainsh_dir/s.$$
printf 'plainsh-session\t1\t%s\n' "$$" > "$__plainsh_session" 2>/dev/null || return 0
__plainsh_seq=0 __plainsh_fails=0 __plainsh_fresh=1 __plainsh_histnum= __plainsh_offer= __plainsh_request= __plainsh_after=
__plainsh_hist=() __plainsh_cmds=() __plainsh_risks=() __plainsh_confs=() __plainsh_reads=()
__plainsh_qfd= __plainsh_rfd=
__plainsh_utf=0; [[ ${LC_ALL:-${LC_CTYPE:-${LANG-}}} == *[Uu][Tt][Ff]* ]] && __plainsh_utf=1

# ---- the session file: every command that runs, and what was run instead of an offer
__plainsh_esc() { REPLY=${1//\\/\\\\}; REPLY=${REPLY//$'\t'/\\t}; REPLY=${REPLY//$'\n'/\\n}; }

if (( BASH_VERSINFO[0] > 5 || (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] >= 3) )); then
  eval '__plainsh_last() { REPLY=${ HISTTIMEFORMAT= builtin history 1; }; }'        # no subshell on bash 5.3+
else
  __plainsh_last() { REPLY=$(HISTTIMEFORMAT= builtin history 1); }
fi

__plainsh_precmd() {
  local st=$? now cmd num dir
  __plainsh_fresh=1
  __plainsh_last
  if [[ -z $__plainsh_histnum ]]; then                     # first prompt of the session: nothing has been run here yet
    [[ $REPLY =~ ^[[:space:]]*([0-9]+) ]] && __plainsh_histnum=${BASH_REMATCH[1]} || __plainsh_histnum=0
  elif [[ $REPLY =~ ^[[:space:]]*([0-9]+)\*?[[:space:]]+(.*)$ ]]; then
    num=${BASH_REMATCH[1]} cmd=${BASH_REMATCH[2]}
    if [[ $num != "$__plainsh_histnum" ]]; then
      if :; then
        printf -v now '%(%s)T' -1
        __plainsh_esc "$PWD"; dir=$REPLY
        __plainsh_esc "$cmd"
        printf 'run\t%s\t%s\t%s\t%s\n' "$now" "$st" "$dir" "$REPLY" >> "$__plainsh_session"
        if [[ -n $__plainsh_offer && $cmd != "$__plainsh_offer" ]]; then
          local ran=$REPLY
          __plainsh_esc "$__plainsh_request"; local req=$REPLY
          __plainsh_esc "$__plainsh_offer"
          printf 'fix\t%s\t%s\t%s\n' "$req" "$REPLY" "$ran" >> "$__plainsh_session"
        fi
      fi
      __plainsh_histnum=$num
    fi
  fi
  __plainsh_offer=
  return $st
}

# ---- talking to the answering process (started on demand, shared by all terminals, exits when idle)
__plainsh_serve() {
  local pid
  if [[ -p $__plainsh_dir/req && -r $__plainsh_dir/serve.pid ]]; then
    read -r pid < "$__plainsh_dir/serve.pid"
    [[ -n $pid ]] && kill -0 "$pid" 2>/dev/null && return 0
  fi
  ( "$__plainsh_bin" --serve & ) >/dev/null 2>&1
}

__plainsh_open() {
  local r=$__plainsh_dir/r.$$ i
  if [[ -z $__plainsh_rfd ]]; then
    [[ -p $r ]] || mkfifo -m 600 "$r" 2>/dev/null || return 1
    exec {__plainsh_rfd}<>"$r" || { __plainsh_rfd=; return 1; }
  fi
  if [[ -z $__plainsh_qfd ]]; then
    for i in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do      # the pipe appears when the process is up
      [[ -p $__plainsh_dir/req ]] && break
      read -r -t 0.05 -u "$__plainsh_rfd" _ 2>/dev/null
    done
    [[ -p $__plainsh_dir/req ]] || return 1
    exec {__plainsh_qfd}<>"$__plainsh_dir/req" || { __plainsh_qfd=; return 1; }
  fi
}

# __plainsh_ask MODE TEXT [TIMEOUT]  ->  fills __plainsh_cmds/risks/confs/reads; 0 = answered (possibly with nothing)
__plainsh_ask() {
  local seq=$(( ++__plainsh_seq )) line parts i n tries=0
  __plainsh_cmds=() __plainsh_risks=() __plainsh_confs=() __plainsh_reads=()
  __plainsh_serve
  __plainsh_open || return 1
  printf '%s\x1f%s\x1f%s\x1f%s\x1f%s\x1f%s\n' "$seq" "$__plainsh_dir/r.$$" "$1" "$__plainsh_session" "$PWD" "$2" >&"$__plainsh_qfd" || return 1
  while IFS= read -r -t "${3:-0.5}" -u "$__plainsh_rfd" line; do
    if [[ ${line%%$'\x1f'*} == "$seq" ]]; then
      IFS=$'\x1f' read -r -a parts <<< "$line"
      n=${parts[1]:-0}
      for (( i = 0; i < n; i++ )); do
        __plainsh_cmds[i]=${parts[2 + 4 * i]//$'\x1e'/$'\n'}
        __plainsh_risks[i]=${parts[3 + 4 * i]}
        __plainsh_confs[i]=${parts[4 + 4 * i]}
        __plainsh_reads[i]=${parts[5 + 4 * i]}
      done
      return 0
    fi
    (( ++tries > 50 )) && break
  done
  return 1
}

# a final answer, falling back to a one-off run of the program if the answering process is not reachable
__plainsh_final() {
  if __plainsh_ask "final:${2:-0}" "$1" 3; then __plainsh_fails=0; return 0; fi
  local out
  out=$("$__plainsh_bin" --shell --session "$__plainsh_session" "$1" 2>/dev/null) || return 0
  { IFS= read -r '__plainsh_cmds[0]'; IFS= read -r '__plainsh_risks[0]'; IFS= read -r '__plainsh_confs[0]'; IFS= read -r '__plainsh_reads[0]'; } <<< "$out"
}

# ---- drawing. These use the editor's local variables (bash looks names up in the caller).
__plainsh_draw() {
  local out=$'\e[?2026h' cols=${COLUMNS:-80} n=0 line total last cpos crow ccol up
  (( row > 0 )) && out+=$'\e['"$row"'A'
  out+=$'\r\e[J'"$shown"$'\e['"$color"'m'"$buf"$'\e[0m'
  total=$(( pw + ${#buf} ))
  last=$(( total > 0 ? (total - 1) / cols : 0 ))
  for line in "${pv[@]}"; do out+=$'\r\n'"$line"; (( n++ )); done
  cpos=$(( pw + cur )) crow=$(( cpos / cols )) ccol=$(( cpos % cols ))
  if (( crow > last + n )); then out+=$'\r\n'; (( n++ )); fi
  up=$(( last + n - crow ))
  (( up > 0 )) && out+=$'\e['"$up"'A'
  out+=$'\r'
  (( ccol > 0 )) && out+=$'\e['"$ccol"'C'
  row=$crow
  printf '%s\e[?2026l' "$out"
}

__plainsh_preview() {            # build the lines shown under the request from the current answer
  local cols=$(( ${COLUMNS:-80} - 1 )) n=${#__plainsh_cmds[@]} cmd why pct arrow='->' more=
  pv=()
  (( __plainsh_utf )) && arrow=$'→'
  if (( n == 0 )); then
    [[ -n ${1-} ]] && pv[0]=$'\e[2m'"  $1"$'\e[0m'
    return
  fi
  (( sel >= n )) && sel=0
  cmd=${__plainsh_cmds[sel]//$'\n'/ ; } why=${__plainsh_reads[sel]}
  pct=${__plainsh_confs[sel]/./}; pct=$(( 10#${pct:-0} / 10 ))
  (( n > 1 )) && more="   Tab: $(( sel + 1 ))/$n"
  [[ -z $cmd ]] && cmd="(nothing to run)"
  (( ${#cmd} > cols - 5 )) && cmd=${cmd:0:cols - 8}...
  why="$why   $pct%$more"
  (( ${#why} > cols - 5 )) && why=${why:0:cols - 8}...
  if [[ ${__plainsh_risks[sel]} == 2 ]]; then
    pv[0]=$'\e[1;31m'"  $arrow $cmd"$'\e[0m'
  else
    pv[0]=$'\e[1m'"  $arrow $cmd"$'\e[0m'
  fi
  if (( pct < 75 )); then pv[1]=$'\e[33m'"    not sure: $why"$'\e[0m'; else pv[1]=$'\e[2m'"    $why"$'\e[0m'; fi
}

__plainsh_say() {                # the lines left on screen above the offered command
  local pct=${__plainsh_confs[0]/./}; pct=$(( 10#${pct:-0} / 10 ))
  if (( pct < 75 )); then
    printf '\r\n\e[33m  not sure (%d%%): %s\e[0m' "$pct" "${__plainsh_reads[0]}"
  else
    printf '\r\n\e[2m  %s  (%d%% sure)\e[0m' "${__plainsh_reads[0]}" "$pct"
  fi
  [[ ${__plainsh_risks[0]} == 2 ]] && printf '\r\n\e[1;31m  this removes or stops things: read it before pressing Enter\e[0m'
  return 0
}

__plainsh_insert() { buf=${buf:0:cur}$1${buf:cur}; cur=$(( cur + ${#1} )); }
__plainsh_word_left() { while (( cur > 1 )) && [[ ${buf:cur-1:1} == ' ' ]]; do (( cur-- )); done; while (( cur > 1 )) && [[ ${buf:cur-1:1} != ' ' ]]; do (( cur-- )); done; }
__plainsh_word_right() { local n=${#buf}; while (( cur < n )) && [[ ${buf:cur:1} == ' ' ]]; do (( cur++ )); done; while (( cur < n )) && [[ ${buf:cur:1} != ' ' ]]; do (( cur++ )); done; }

# ---- the request line editor: runs from the moment ":" is typed on an empty prompt until Enter or Esc
__plainsh_editor() {
  local buf=: cur=1 row=0 sel=0 leave=0 key k2 seq rc dirty=0 hi=${#__plainsh_hist[@]} draft= text old tty paste
  local color=${PLAINSH_COLOR:-38;5;141} live=${PLAINSH_PREVIEW:-1}
  local -a pv=()
  local prompt=${PS1@P}
  prompt=${prompt##*$'\n'}
  local shown=${prompt//[$'\001\002']/} vis=$prompt
  while [[ $vis == *$'\001'*$'\002'* ]]; do vis=${vis%%$'\001'*}${vis#*$'\002'}; done
  local pw=${#vis}
  tty=$(stty -g 2>/dev/null) || return 1
  stty -isig -ixon 2>/dev/null
  __plainsh_cmds=()
  (( __plainsh_fails >= 3 )) && live=0
  [[ $live == 1 ]] && __plainsh_serve
  __plainsh_draw
  __plainsh_after=keep
  while :; do
    if (( dirty )) && [[ $live == 1 ]]; then
      IFS= read -rsN1 -t 0.07 key; rc=$?
      if (( rc > 128 )); then                              # a pause in typing: show what the request would become
        dirty=0
        if [[ $buf =~ ^:[[:space:]]+([^[:space:]].*)$ ]]; then
          text=${BASH_REMATCH[1]}
          if __plainsh_ask preview "$text" 0.6; then
            __plainsh_fails=0
            if [[ $text == *' '*' '* ]]; then __plainsh_preview "nothing I can do with that yet"; else __plainsh_preview; fi
          else
            pv=()
            (( ++__plainsh_fails >= 3 )) && live=0          # no answers are coming: stop asking while typing
          fi
        else
          pv=()
        fi
        __plainsh_draw
        continue
      fi
    else
      IFS= read -rsN1 key || { key=$'\x03'; }
    fi
    old=$buf
    case $key in
      $'\r'|$'\n')
        if [[ $buf =~ ^:[[:space:]]+([[:alnum:]\"\'~./].*)$ ]]; then
          text=${BASH_REMATCH[1]}
          __plainsh_final "$text" "$sel"
          if (( ${#__plainsh_cmds[@]} == 0 )); then
            sel=0; __plainsh_preview "that is outside what I can do so far: reword it, or press Esc"; __plainsh_draw
            continue
          fi
          __plainsh_hist+=("$text")
          pv=(); cur=${#buf}; __plainsh_draw
          __plainsh_say
          printf '\r\n'
          READLINE_LINE=${__plainsh_cmds[0]} READLINE_POINT=100000
          __plainsh_offer=${__plainsh_cmds[0]} __plainsh_request=$text
        else                                               # a real ":" command such as ": > file": run it as typed
          pv=(); cur=${#buf}; __plainsh_draw
          (( row > 0 )) && printf '\e[%dA' "$row"
          printf '\r\e[J'
          READLINE_LINE=$buf READLINE_POINT=100000
          __plainsh_after=accept
        fi
        break ;;
      $'\x7f'|$'\b')                                       # Backspace; over the colon itself it leaves
        if (( cur > 1 )); then buf=${buf:0:cur-1}${buf:cur}; (( cur-- )); elif (( ${#buf} == 1 )); then leave=1; fi ;;
      $'\x03'|$'\x07') leave=1 ;;                          # Ctrl-C, Ctrl-G
      $'\x04') if (( ${#buf} == 1 )); then leave=1; else buf=${buf:0:cur}${buf:cur+1}; fi ;;
      $'\x01') cur=1 ;;
      $'\x05') cur=${#buf} ;;
      $'\x02') (( cur > 1 )) && (( cur-- )) ;;
      $'\x06') (( cur < ${#buf} )) && (( cur++ )) ;;
      $'\x15') buf=:${buf:cur}; cur=1 ;;
      $'\x0b') buf=${buf:0:cur} ;;
      $'\x17') seq=$cur; __plainsh_word_left; buf=${buf:0:cur}${buf:seq} ;;
      $'\x0c') printf '\e[H\e[2J'; row=0 ;;
      $'\t') (( ${#__plainsh_cmds[@]} > 1 )) && { sel=$(( (sel + 1) % ${#__plainsh_cmds[@]} )); __plainsh_preview; } ;;
      $'\e')
        if ! IFS= read -rsN1 -t 0.03 k2; then leave=1                      # Esc on its own
        elif [[ $k2 == '[' || $k2 == O ]]; then
          seq=$k2
          while IFS= read -rsN1 -t 0.03 k2; do seq+=$k2; [[ $k2 == [@-~] ]] && break; done
          case $seq in
            '[A'|OA) if (( hi > 0 )); then (( hi == ${#__plainsh_hist[@]} )) && draft=$buf; (( hi-- )); buf=": ${__plainsh_hist[hi]}"; cur=${#buf}; fi ;;
            '[B'|OB) if (( hi < ${#__plainsh_hist[@]} )); then (( hi++ )); if (( hi == ${#__plainsh_hist[@]} )); then buf=$draft; else buf=": ${__plainsh_hist[hi]}"; fi; cur=${#buf}; fi ;;
            '[C'|OC) (( cur < ${#buf} )) && (( cur++ )) ;;
            '[D'|OD) (( cur > 1 )) && (( cur-- )) ;;
            '[H'|OH|'[1~'|'[7~') cur=1 ;;
            '[F'|OF|'[4~'|'[8~') cur=${#buf} ;;
            '[3~') buf=${buf:0:cur}${buf:cur+1} ;;
            '[1;5C'|'[1;3C') __plainsh_word_right ;;
            '[1;5D'|'[1;3D') __plainsh_word_left ;;
            '[200~')                                       # pasted text: take it as typed, on one line
              paste=
              while IFS= read -rsN1 -t 1 k2; do paste+=$k2; [[ $paste == *$'\e[201~' ]] && break; done
              paste=${paste%$'\e[201~'}; paste=${paste//[$'\r\n\t']/ }
              __plainsh_insert "$paste" ;;
          esac
        elif [[ $k2 == b ]]; then __plainsh_word_left
        elif [[ $k2 == f ]]; then __plainsh_word_right
        elif [[ $k2 == $'\x7f' ]]; then seq=$cur; __plainsh_word_left; buf=${buf:0:cur}${buf:seq}
        fi ;;
      [[:cntrl:]]) ;;
      *) __plainsh_insert "$key" ;;
    esac
    if (( leave )); then
      (( row > 0 )) && printf '\e[%dA' "$row"
      printf '\r\e[J'
      READLINE_LINE= READLINE_POINT=0
      break
    fi
    if [[ $buf != "$old" ]]; then dirty=1; sel=0; fi
    __plainsh_draw
  done
  stty "$tty" 2>/dev/null
  return 0
}

# ---- keys
# ":" on an empty prompt opens the editor; anywhere else it is an ordinary colon.
__plainsh_colon() {
  if [[ -z $READLINE_LINE && $__plainsh_fresh == 1 ]] && __plainsh_editor; then
    if [[ $__plainsh_after == accept ]]; then
      __plainsh_fresh=0
      bind '"\e[90012~": accept-line'
    else
      bind '"\e[90012~": redraw-current-line'
    fi
  else
    bind '"\e[90012~": "\e[90013~:"'
  fi
}

# Enter on a line that already starts with ": " (pasted, or recalled from history) is translated the same way.
__plainsh_enter() {
  local line=$READLINE_LINE
  if [[ $line =~ ^:[[:space:]]+([[:alnum:]\"\'~./].*)$ ]]; then
    local text=${BASH_REMATCH[1]}
    __plainsh_final "$text" 0
    if (( ${#__plainsh_cmds[@]} == 0 )); then
      printf '\n  plainsh: that is outside what I can do so far\n'
    else
      __plainsh_say; printf '\r\n'
      __plainsh_hist+=("$text")
      READLINE_LINE=${__plainsh_cmds[0]} READLINE_POINT=100000
      __plainsh_offer=${__plainsh_cmds[0]} __plainsh_request=$text
    fi
    bind '"\e[90015~": redraw-current-line'
  else
    __plainsh_fresh=0
    bind '"\e[90015~": accept-line'
  fi
}

if [[ ${PROMPT_COMMAND-} != *__plainsh_precmd* ]]; then
  if [[ $(declare -p PROMPT_COMMAND 2>/dev/null) == "declare -a"* ]]; then
    PROMPT_COMMAND=(__plainsh_precmd "${PROMPT_COMMAND[@]}")
  else
    PROMPT_COMMAND="__plainsh_precmd${PROMPT_COMMAND:+;$PROMPT_COMMAND}"
  fi
fi

for __plainsh_map in emacs-standard vi-insert; do
  bind -m "$__plainsh_map" '"\e[90013~": quoted-insert'
  bind -m "$__plainsh_map" '"\e[90012~": "\e[90013~:"'
  bind -m "$__plainsh_map" '"\e[90015~": accept-line'
  bind -m "$__plainsh_map" -x '"\e[90011~": __plainsh_colon'
  bind -m "$__plainsh_map" -x '"\e[90014~": __plainsh_enter'
  bind -m "$__plainsh_map" '":": "\e[90011~\e[90012~"'
  bind -m "$__plainsh_map" '"\C-m": "\e[90014~\e[90015~"'
  bind -m "$__plainsh_map" '"\C-j": "\e[90014~\e[90015~"'
done
unset __plainsh_map
