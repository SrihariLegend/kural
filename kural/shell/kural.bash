# kural for bash: type ":" then plain English.
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
# Every answer is one short run of the `kural` binary (KURAL_BIN, default "kural"); nothing stays running.
#
# Settings (set before sourcing):  KURAL_COLOR  colour of the request line, as SGR numbers (default 38;5;141)
#                                  KURAL_PREVIEW=0  no live preview while typing

[[ $- == *i* && -n ${BASH_VERSION-} ]] || return 0
(( BASH_VERSINFO[0] > 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] >= 4) )) || return 0
__kural_bin=${KURAL_BIN:-kural}
command -v "$__kural_bin" >/dev/null 2>&1 || return 0

if [[ -n ${XDG_RUNTIME_DIR-} && -d $XDG_RUNTIME_DIR ]]; then __kural_dir=$XDG_RUNTIME_DIR/kural; else __kural_dir=/tmp/kural-$UID; fi
[[ -d $__kural_dir ]] || mkdir -p -m 700 "$__kural_dir" 2>/dev/null || return 0
for __kural_f in "$__kural_dir"/s.* "$__kural_dir"/r.*; do               # files left by terminals that are gone
  [[ -e $__kural_f && ${__kural_f##*.} =~ ^[0-9]+$ ]] || continue
  kill -0 "${__kural_f##*.}" 2>/dev/null || rm -f -- "$__kural_f"
done
unset __kural_f
__kural_session=$__kural_dir/s.$$
printf 'kural-session\t1\t%s\n' "$$" > "$__kural_session" 2>/dev/null || return 0
__kural_fails=0 __kural_fresh=1 __kural_histnum= __kural_offer= __kural_request= __kural_after=
__kural_hist=() __kural_cmds=() __kural_risks=() __kural_confs=() __kural_reads=()
__kural_utf=0; [[ ${LC_ALL:-${LC_CTYPE:-${LANG-}}} == *[Uu][Tt][Ff]* ]] && __kural_utf=1

# ---- the session file: every command that runs, and what was run instead of an offer
__kural_esc() { REPLY=${1//\\/\\\\}; REPLY=${REPLY//$'\t'/\\t}; REPLY=${REPLY//$'\n'/\\n}; }

if (( BASH_VERSINFO[0] > 5 || (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] >= 3) )); then
  eval '__kural_last() { REPLY=${ HISTTIMEFORMAT= builtin history 1; }; }'        # no subshell on bash 5.3+
else
  __kural_last() { REPLY=$(HISTTIMEFORMAT= builtin history 1); }
fi

__kural_precmd() {
  local st=$? now cmd num dir
  __kural_fresh=1
  __kural_last
  if [[ -z $__kural_histnum ]]; then                     # first prompt of the session: nothing has been run here yet
    [[ $REPLY =~ ^[[:space:]]*([0-9]+) ]] && __kural_histnum=${BASH_REMATCH[1]} || __kural_histnum=0
  elif [[ $REPLY =~ ^[[:space:]]*([0-9]+)\*?[[:space:]]+(.*)$ ]]; then
    num=${BASH_REMATCH[1]} cmd=${BASH_REMATCH[2]}
    if [[ $num != "$__kural_histnum" ]]; then
      if :; then
        printf -v now '%(%s)T' -1
        __kural_esc "$PWD"; dir=$REPLY
        __kural_esc "$cmd"
        printf 'run\t%s\t%s\t%s\t%s\n' "$now" "$st" "$dir" "$REPLY" >> "$__kural_session"
        if [[ -n $__kural_offer && $cmd != "$__kural_offer" ]]; then
          local ran=$REPLY
          __kural_esc "$__kural_request"; local req=$REPLY
          __kural_esc "$__kural_offer"
          printf 'fix\t%s\t%s\t%s\n' "$req" "$REPLY" "$ran" >> "$__kural_session"
        fi
      fi
      __kural_histnum=$num
    fi
  fi
  __kural_offer=
  return $st
}

# ---- asking the kural binary: one short call per request, nothing is left running
# __kural_ask MODE TEXT [CHOOSE]  ->  fills __kural_cmds/risks/confs/reads; 0 = answered (possibly with nothing)
#   MODE is "preview" (while typing: nothing is recorded) or "final" (Enter: the offer is recorded in the session file)
__kural_ask() {
  local line i n
  local -a parts flags=(--hook --session "$__kural_session" --cwd "$PWD")
  [[ $1 == preview ]] && flags+=(--preview)
  (( ${3:-0} > 0 )) && flags+=(--choose "$3")
  __kural_cmds=() __kural_risks=() __kural_confs=() __kural_reads=()
  line=$("$__kural_bin" "${flags[@]}" -- "$2" 2>/dev/null) || return 1
  IFS=$'\x1f' read -r -a parts <<< "$line"
  n=${parts[0]:-x}
  [[ $n =~ ^[0-9]+$ ]] || return 1
  for (( i = 0; i < n; i++ )); do
    __kural_cmds[i]=${parts[1 + 4 * i]//$'\x1e'/$'\n'}
    __kural_risks[i]=${parts[2 + 4 * i]}
    __kural_confs[i]=${parts[3 + 4 * i]}
    __kural_reads[i]=${parts[4 + 4 * i]}
  done
  return 0
}

__kural_final() {
  __kural_ask final "$1" "${2:-0}" && __kural_fails=0
  return 0
}

# ---- drawing. These use the editor's local variables (bash looks names up in the caller).
__kural_draw() {
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

__kural_preview() {            # build the lines shown under the request from the current answer
  local cols=$(( ${COLUMNS:-80} - 1 )) n=${#__kural_cmds[@]} cmd why pct arrow='->' more=
  pv=()
  (( __kural_utf )) && arrow=$'→'
  if (( n == 0 )); then
    [[ -n ${1-} ]] && pv[0]=$'\e[2m'"  $1"$'\e[0m'
    return
  fi
  (( sel >= n )) && sel=0
  cmd=${__kural_cmds[sel]//$'\n'/ ; } why=${__kural_reads[sel]}
  pct=${__kural_confs[sel]/./}; pct=$(( 10#${pct:-0} / 10 ))
  (( n > 1 )) && more="   Tab: $(( sel + 1 ))/$n"
  [[ -z $cmd ]] && cmd="(nothing to run)"
  (( ${#cmd} > cols - 5 )) && cmd=${cmd:0:cols - 8}...
  why="$why   $pct%$more"
  (( ${#why} > cols - 5 )) && why=${why:0:cols - 8}...
  if [[ ${__kural_risks[sel]} == 2 ]]; then
    pv[0]=$'\e[1;31m'"  $arrow $cmd"$'\e[0m'
  else
    pv[0]=$'\e[1m'"  $arrow $cmd"$'\e[0m'
  fi
  if (( pct < 75 )); then pv[1]=$'\e[33m'"    not sure: $why"$'\e[0m'; else pv[1]=$'\e[2m'"    $why"$'\e[0m'; fi
}

__kural_say() {                # the lines left on screen above the offered command
  local pct=${__kural_confs[0]/./}; pct=$(( 10#${pct:-0} / 10 ))
  if (( pct < 75 )); then
    printf '\r\n\e[33m  not sure (%d%%): %s\e[0m' "$pct" "${__kural_reads[0]}"
  else
    printf '\r\n\e[2m  %s  (%d%% sure)\e[0m' "${__kural_reads[0]}" "$pct"
  fi
  [[ ${__kural_risks[0]} == 2 ]] && printf '\r\n\e[1;31m  this removes or stops things: read it before pressing Enter\e[0m'
  return 0
}

__kural_insert() { buf=${buf:0:cur}$1${buf:cur}; cur=$(( cur + ${#1} )); }
__kural_word_left() { while (( cur > 1 )) && [[ ${buf:cur-1:1} == ' ' ]]; do (( cur-- )); done; while (( cur > 1 )) && [[ ${buf:cur-1:1} != ' ' ]]; do (( cur-- )); done; }
__kural_word_right() { local n=${#buf}; while (( cur < n )) && [[ ${buf:cur:1} == ' ' ]]; do (( cur++ )); done; while (( cur < n )) && [[ ${buf:cur:1} != ' ' ]]; do (( cur++ )); done; }

# ---- the request line editor: runs from the moment ":" is typed on an empty prompt until Enter or Esc
__kural_editor() {
  local buf=: cur=1 row=0 sel=0 leave=0 key k2 seq rc dirty=0 hi=${#__kural_hist[@]} draft= text old tty paste
  local color=${KURAL_COLOR:-38;5;141} live=${KURAL_PREVIEW:-1}
  local -a pv=()
  local prompt=${PS1@P}
  prompt=${prompt##*$'\n'}
  local shown=${prompt//[$'\001\002']/} vis=$prompt
  while [[ $vis == *$'\001'*$'\002'* ]]; do vis=${vis%%$'\001'*}${vis#*$'\002'}; done
  local pw=${#vis}
  tty=$(stty -g 2>/dev/null) || return 1
  stty -isig -ixon 2>/dev/null
  __kural_cmds=()
  (( __kural_fails >= 3 )) && live=0
  __kural_draw
  __kural_after=keep
  while :; do
    if (( dirty )) && [[ $live == 1 ]]; then
      IFS= read -rsN1 -t 0.07 key; rc=$?
      if (( rc > 128 )); then                              # a pause in typing: show what the request would become
        dirty=0
        if [[ $buf =~ ^:[[:space:]]+([^[:space:]].*)$ ]]; then
          text=${BASH_REMATCH[1]}
          if __kural_ask preview "$text"; then
            __kural_fails=0
            if [[ $text == *' '*' '* ]]; then __kural_preview "nothing I can do with that yet"; else __kural_preview; fi
          else
            pv=()
            (( ++__kural_fails >= 3 )) && live=0          # no answers are coming: stop asking while typing
          fi
        else
          pv=()
        fi
        __kural_draw
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
          __kural_final "$text" "$sel"
          if (( ${#__kural_cmds[@]} == 0 )); then
            sel=0; __kural_preview "that is outside what I can do so far: reword it, or press Esc"; __kural_draw
            continue
          fi
          __kural_hist+=("$text")
          pv=(); cur=${#buf}; __kural_draw
          __kural_say
          printf '\r\n'
          READLINE_LINE=${__kural_cmds[0]} READLINE_POINT=100000
          __kural_offer=${__kural_cmds[0]} __kural_request=$text
        else                                               # a real ":" command such as ": > file": run it as typed
          pv=(); cur=${#buf}; __kural_draw
          (( row > 0 )) && printf '\e[%dA' "$row"
          printf '\r\e[J'
          READLINE_LINE=$buf READLINE_POINT=100000
          __kural_after=accept
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
      $'\x17') seq=$cur; __kural_word_left; buf=${buf:0:cur}${buf:seq} ;;
      $'\x0c') printf '\e[H\e[2J'; row=0 ;;
      $'\t') (( ${#__kural_cmds[@]} > 1 )) && { sel=$(( (sel + 1) % ${#__kural_cmds[@]} )); __kural_preview; } ;;
      $'\e')
        if ! IFS= read -rsN1 -t 0.03 k2; then leave=1                      # Esc on its own
        elif [[ $k2 == '[' || $k2 == O ]]; then
          seq=$k2
          while IFS= read -rsN1 -t 0.03 k2; do seq+=$k2; [[ $k2 == [@-~] ]] && break; done
          case $seq in
            '[A'|OA) if (( hi > 0 )); then (( hi == ${#__kural_hist[@]} )) && draft=$buf; (( hi-- )); buf=": ${__kural_hist[hi]}"; cur=${#buf}; fi ;;
            '[B'|OB) if (( hi < ${#__kural_hist[@]} )); then (( hi++ )); if (( hi == ${#__kural_hist[@]} )); then buf=$draft; else buf=": ${__kural_hist[hi]}"; fi; cur=${#buf}; fi ;;
            '[C'|OC) (( cur < ${#buf} )) && (( cur++ )) ;;
            '[D'|OD) (( cur > 1 )) && (( cur-- )) ;;
            '[H'|OH|'[1~'|'[7~') cur=1 ;;
            '[F'|OF|'[4~'|'[8~') cur=${#buf} ;;
            '[3~') buf=${buf:0:cur}${buf:cur+1} ;;
            '[1;5C'|'[1;3C') __kural_word_right ;;
            '[1;5D'|'[1;3D') __kural_word_left ;;
            '[200~')                                       # pasted text: take it as typed, on one line
              paste=
              while IFS= read -rsN1 -t 1 k2; do paste+=$k2; [[ $paste == *$'\e[201~' ]] && break; done
              paste=${paste%$'\e[201~'}; paste=${paste//[$'\r\n\t']/ }
              __kural_insert "$paste" ;;
          esac
        elif [[ $k2 == b ]]; then __kural_word_left
        elif [[ $k2 == f ]]; then __kural_word_right
        elif [[ $k2 == $'\x7f' ]]; then seq=$cur; __kural_word_left; buf=${buf:0:cur}${buf:seq}
        fi ;;
      [[:cntrl:]]) ;;
      *) __kural_insert "$key" ;;
    esac
    if (( leave )); then
      (( row > 0 )) && printf '\e[%dA' "$row"
      printf '\r\e[J'
      READLINE_LINE= READLINE_POINT=0
      break
    fi
    if [[ $buf != "$old" ]]; then dirty=1; sel=0; fi
    __kural_draw
  done
  stty "$tty" 2>/dev/null
  return 0
}

# ---- keys
# ":" on an empty prompt opens the editor; anywhere else it is an ordinary colon.
__kural_colon() {
  if [[ -z $READLINE_LINE && $__kural_fresh == 1 ]] && __kural_editor; then
    if [[ $__kural_after == accept ]]; then
      __kural_fresh=0
      bind '"\e[90012~": accept-line'
    else
      bind '"\e[90012~": redraw-current-line'
    fi
  else
    bind '"\e[90012~": "\e[90013~:"'
  fi
}

# Enter on a line that already starts with ": " (pasted, or recalled from history) is translated the same way.
__kural_enter() {
  local line=$READLINE_LINE
  if [[ $line =~ ^:[[:space:]]+([[:alnum:]\"\'~./].*)$ ]]; then
    local text=${BASH_REMATCH[1]}
    __kural_final "$text" 0
    if (( ${#__kural_cmds[@]} == 0 )); then
      printf '\n  kural: that is outside what I can do so far\n'
    else
      __kural_say; printf '\r\n'
      __kural_hist+=("$text")
      READLINE_LINE=${__kural_cmds[0]} READLINE_POINT=100000
      __kural_offer=${__kural_cmds[0]} __kural_request=$text
    fi
    bind '"\e[90015~": redraw-current-line'
  else
    __kural_fresh=0
    bind '"\e[90015~": accept-line'
  fi
}

if [[ ${PROMPT_COMMAND-} != *__kural_precmd* ]]; then
  if [[ $(declare -p PROMPT_COMMAND 2>/dev/null) == "declare -a"* ]]; then
    PROMPT_COMMAND=(__kural_precmd "${PROMPT_COMMAND[@]}")
  else
    PROMPT_COMMAND="__kural_precmd${PROMPT_COMMAND:+;$PROMPT_COMMAND}"
  fi
fi

for __kural_map in emacs-standard vi-insert; do
  bind -m "$__kural_map" '"\e[90013~": quoted-insert'
  bind -m "$__kural_map" '"\e[90012~": "\e[90013~:"'
  bind -m "$__kural_map" '"\e[90015~": accept-line'
  bind -m "$__kural_map" -x '"\e[90011~": __kural_colon'
  bind -m "$__kural_map" -x '"\e[90014~": __kural_enter'
  bind -m "$__kural_map" '":": "\e[90011~\e[90012~"'
  bind -m "$__kural_map" '"\C-m": "\e[90014~\e[90015~"'
  bind -m "$__kural_map" '"\C-j": "\e[90014~\e[90015~"'
done
unset __kural_map
