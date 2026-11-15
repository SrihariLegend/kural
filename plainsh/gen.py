"""Synthetic training data: random valid programs, each paired with a plain-English request.

The program side is sampled from the typed language in ir.py, so it is always correct by
construction. The English side is assembled from many interchangeable phrasings, word orders,
politeness wrappers and value spellings. Argument values are random, so the model has to learn
to point at them rather than memorise them.
"""
import random

from . import ir, tok

# --------------------------------------------------------------------------- value pools
WORDS = """alpha beta gamma delta report draft notes invoice budget summary backup archive photo holiday
project client server config settings data dataset results output input cache temp test tests demo sample
old new final copy scratch thesis paper slides resume letter contract receipt statement export import
log logs error debug trace metrics users orders products sales inventory schedule plan roadmap meeting
music video movie clip song album picture image screenshot wallpaper icon font theme script tool utils
helpers lib core main app api web site blog docs readme license todo changelog release build dist
module package plugin widget model weights train eval run job task queue worker daemon service kernel
alice bob carol dave erin frank grace heidi ivan judy mallory oscar peggy trent victor walter
january february march april june july august september october november december monday friday
red green blue black white silver gold tiny huge fast slow sharp flat round square north south east west
apple banana cherry mango lemon peach olive maple cedar pine birch willow river lake ocean island
falcon eagle sparrow otter badger tiger panda koala lynx heron comet nova pulsar quasar nebula orbit""".split()
EXTS = "py js ts rs go c h cpp java rb sh md txt log csv json yaml yml xml html css pdf jpg jpeg png gif svg mp3 mp4 mkv mov wav flac zip tar gz bak tmp conf ini toml lock sql db iso deb exe docx xlsx".split()
EXT_NAMES = ["python", "javascript", "typescript", "rust", "markdown", "text", "shell", "ruby", "golang", "java",
             "json", "yaml", "csv", "html", "css", "log", "pdf", "backup", "temp", "config", "c++", "kotlin"]
EXT_PLURALS = ["pdfs", "logs", "images", "pictures", "photos", "videos", "movies", "songs", "archives", "zips", "jpgs",
               "pngs", "mp4s", "mp3s", "csvs", "isos", "gifs", "docs" if False else "txts", "jsons", "backups"]
FOLDER_NAMES = ["downloads", "documents", "desktop", "pictures", "music", "videos", "home", "tmp", "projects", "trash",
                "photos", "movies", "downloads", "documents", "desktop"]
REAL_DIRS = ["~/Downloads", "~/Documents", "/var/log", "/etc", "/tmp", "~", "/usr/share", "~/.config", "/opt",
             "/home", "/var", "~/Desktop", "/usr/local/bin", "node_modules", "src", "tests", "build", "dist",
             "docs", "./build", "../data", "~/code", "/mnt/usb", "/srv/www", "~/.cache", "lib", "bin", "target"]
DIR_WORDS = """assets packages www public static vendor scripts config data images media uploads logs cache backups
notes work dev code repos include templates migrations fixtures resources themes plugins modules components pages
styles fonts icons locales samples examples benchmarks tools utils archive old tmp temp output results reports
exports imports shared common core api server client frontend backend mobile desktop web app apps services""".split()
SPECIAL_FILES = ["/etc/hosts", "/etc/fstab", "/etc/passwd", "Makefile", "Dockerfile", "README", "LICENSE", ".bashrc", "~/.bashrc",
                 "~/.zshrc", ".gitignore", "/proc/cpuinfo", "/etc/os-release", ".env", "~/.ssh/config", "/var/log/syslog",
                 "/var/log/messages", "Cargo.toml", "package.json", "requirements.txt", "CHANGELOG", "TODO", "notes",
                 "/etc/resolv.conf", "~/.profile", "/var/log/auth.log", "docker-compose.yml", "/etc/ssh/sshd_config",
                 "Gemfile", "go.mod", "pyproject.toml", "/etc/hostname", "~/.gitconfig", "/proc/meminfo", "Vagrantfile"]
PKGS = """htop vim neovim git curl wget tmux fzf ripgrep bat fd jq tree rsync docker nodejs npm python-pip gcc make
cmake clang rustup go ffmpeg imagemagick gimp vlc mpv firefox chromium thunderbird libreoffice inkscape blender
nginx postgresql redis sqlite zsh fish neofetch btop ncdu unzip p7zip zip openssh nmap wireshark strace gdb
valgrind kdenlive obs-studio steam discord telegram-desktop qbittorrent keepassxc syncthing restic borg""".split()
PROCS = """firefox chrome chromium node python python3 java nginx postgres redis docker code slack discord steam
spotify zoom vlc mpv bash zsh ssh sshd cron systemd plasmashell kwin electron thunderbird obs blender gimp
webpack cargo rustc gcc make npm yarn ruby php mysqld mongod telegram signal""".split()
BRANCHES = ["main", "master", "dev", "develop", "staging", "feature/login", "feature/search", "fix-bug-12", "hotfix",
            "release-1.2", "experiment", "refactor", "wip", "testing", "docs-update", "v2", "feat/api", "bugfix/crash"]
PHRASES = ["import numpy", "TODO", "FIXME", "error", "timeout", "password", "api key", "connection refused",
           "segfault", "deprecated", "hello world", "not found", "warning", "fatal", "http://", "localhost",
           "null pointer", "out of memory", "copyright", "license", "main()", "def test", "console.log", "panic",
           "failed", "denied", "exception", "traceback", "debug", "user id", "access token", "retry", "404"]
MSGS = ["fix login bug", "initial commit", "update readme", "add tests", "refactor parser", "wip", "bump version",
        "fix typo", "add search feature", "remove dead code", "improve error handling", "update dependencies",
        "first draft", "cleanup", "handle empty input", "speed up startup", "add dark mode", "fix crash on exit"]
NUM_WORDS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "twelve", "fifteen", "twenty"]
POLITE = ["please", "can you", "could you", "i want to", "i need to", "i'd like to", "help me", "let's", "go ahead and",
          "quickly", "now", "ok", "hey", "pls", "can u", "would you", "just", "i wanna", "time to", "kindly",
          "yo", "ok so", "alright", "need to", "want to", "gotta", "let me", "so"]
TAILS = ["please", "for me", "thanks", "pls", "now", "if you can", "right now", "thank you", "asap", "ok", "quickly"]

UNITS = """nginx sshd docker bluetooth NetworkManager cups cronie postgresql redis mysql apache2 httpd ufw firewalld
tailscaled libvirtd sddm gdm pipewire systemd-resolved fail2ban plex jellyfin mosquitto grafana prometheus ollama
syncthing tor mariadb mongodb elasticsearch rabbitmq caddy haproxy samba nfs-server chronyd avahi-daemon
wpa_supplicant ssh cron apache postgres containerd k3s vsftpd dnsmasq squid zerotier-one transmission""".split()
HOSTS = ["10.0.0.5", "192.168.1.20", "example.com", "myserver", "prod-01", "github.com", "raspberrypi.local", "google.com",
         "8.8.8.8", "pi@192.168.0.12", "root@10.1.2.3", "deploy@staging.example.org", "nas", "db1.internal", "localhost",
         "archlinux.org", "user@host", "srv.example.net", "172.16.0.9", "homelab", "vps", "alice@build-box", "1.1.1.1",
         "wikipedia.org", "my-router", "192.168.0.1", "git@gitlab.com", "backup.lan", "node3", "admin@10.10.10.10"]
URLS = ["https://example.com/file.zip", "http://releases.org/app.tar.gz", "https://github.com/user/repo", "example.org/data.csv",
        "https://github.com/torvalds/linux.git", "git@github.com:me/dotfiles.git", "https://api.example.com/v1/status",
        "http://localhost:8080/health", "https://cdn.site.net/img/logo.png", "https://get.tool.sh/install.sh",
        "https://files.example.io/dataset-2024.tar.xz", "localhost:3000", "https://gitlab.com/group/project.git",
        "http://192.168.1.5:9000/metrics", "https://example.org", "https://dl.example.net/iso/distro-24.04.iso"]
ENVNAMES = """PATH HOME EDITOR JAVA_HOME API_KEY DEBUG NODE_ENV DATABASE_URL LANG TERM SHELL USER PYTHONPATH GOPATH
AWS_PROFILE PORT HOSTNAME XDG_CONFIG_HOME CC LD_LIBRARY_PATH TZ DISPLAY VISUAL OPENAI_API_KEY MY_VAR FOO BAR
SECRET_TOKEN LOG_LEVEL HTTP_PROXY CUDA_VISIBLE_DEVICES RUST_BACKTRACE PAGER BROWSER APP_ENV""".split()
CMDNAMES = """tar ls grep rsync ffmpeg awk sed git docker curl ssh find chmod systemctl python node gcc make cargo npm
pip vim nvim tmux htop rg fd jq go java ruby php kubectl terraform ansible cmake clang zsh bash fish less cat cp mv dd
df du ps top xargs sort uniq cut tr wc python3 rustc deno bun yarn pnpm emacs nano code firefox mpv""".split()
CONTAINERS = ["web", "db", "redis", "nginx", "app", "api", "worker", "my-app", "postgres", "backend", "frontend", "proxy",
              "cache", "3f2a9c1b", "a1b2c3d4e5f6", "mongo", "grafana", "traefik", "test-runner", "dev_db_1", "project-web-1"]
IMAGES = ["nginx", "ubuntu:22.04", "python:3.12", "alpine", "redis:7", "postgres:16", "node:20", "user/app:latest",
          "debian:stable-slim", "golang:1.22", "mysql", "busybox", "ghcr.io/org/tool:v2", "archlinux", "myapp:dev"]
SCRIPTS = "build test start dev lint deploy watch format serve clean typecheck preview e2e docs migrate".split()
SSIDS = ["HomeWifi", "Office 5G", "eduroam", "NETGEAR-42", "Starbucks", "my hotspot", "TP-Link_5F3A", "guest", "Pixel 8",
         "CoffeeShop Free", "FRITZ!Box 7590", "lab-net", "AndroidAP", "Linksys01234"]
OWNERS = ["root", "www-data", "alice", "bob", "nobody", "postgres", "me", "deploy", "http", "git", "myself", "carol",
          "admin", "ubuntu", "dave", "nginx", "mysql", "jenkins"]
MODE_WORDS = ["executable", "read only", "read-only", "writable", "private", "public", "644", "755", "600", "700", "777",
              "400", "u+x", "go-w", "a+r", "+x", "o-rwx", "g+w", "0644", "readonly", "world readable", "664", "750"]
DATES = ["2024-01-15", "jan 5", "last monday", "march 3rd", "last friday", "2023-12-31", "december 1", "june 2024",
         "oct 12 2023", "tuesday", "2025-06-30", "new year", "the 1st of may", "feb 14", "last sunday", "2024-11-01",
         "september 9th", "01/02/2024", "april", "last christmas", "wednesday", "aug 20 2022"]
EXPRS = ["12*8", "1920 / 1080", "2^10", "(4+5)*3", "15 percent of 240", "7 times 6", "100 divided by 7", "3.14 * 2",
         "144/12", "2 plus 2", "1024*1024", "99 minus 45", "5 squared", "17 % 5", "60*60*24", "2 + 2", "365 * 24",
         "1000 / 3", "18 percent of 55", "(12 + 30) / 2", "8 times 125", "45 plus 78 plus 9"]
REMOTES = ["pi@host:/backup", "user@server:~/uploads", "nas:/volume1/backup", "root@10.0.0.5:/srv", "backup-box:",
           "deploy@prod:/var/www", "me@vps:files/", "alice@192.168.1.9:~/incoming", "storage.lan:/data", "myserver:~",
           "ubuntu@ec2-host:/home/ubuntu", "nas:photos/", "web01:/var/www/html", "git@build:/artifacts"]
PCTS = ["50", "50%", "30 percent", "80%", "100", "half", "max", "full", "25%", "10", "75 percent", "0", "60%", "40",
        "90%", "15", "100%", "20 percent", "5"]
PORTS = ["80", "443", "8080", "3000", "5000", "8000", "22", "5432", "3306", "6379", "9090", "4200", "8888", "27017",
         "5173", "9000", "25565", "1234", "8443", "53"]
AFFIXES = ["old_", "new-", "backup_", "2024-", "_final", "-v2", "_old", ".bak", "draft-", "IMG_", "copy_", "_edited",
           "holiday-", "x_", "-small", "_backup", "tmp-", "archive_"]
ALIASES = ["ll", "gs", "k", "la", "gp", "dc", "py", "..", "cls", "gco", "update", "serve", "weather", "please", "tf", "v"]
ALIAS_CMDS = ["ls -la", "git status", "kubectl", "git push", "docker compose", "python3", "cd ..", "clear",
              "git checkout", "sudo pacman -Syu", "python3 -m http.server", "sudo", "terraform", "nvim", "ls -lah --color"]
VAR_VALUES = ["vim", "production", "1", "/usr/lib/jvm/java-17", "true", "development", "8080", "en_US.UTF-8", "nano",
              "debug", "/opt/tools/bin", "abc123", "0", "postgres://localhost/db", "false", "firefox", "42", "eu-west-1"]
SENTENCES = ["hello world", "export EDITOR=vim", "TODO: fix this", "127.0.0.1 myhost", "done", "node_modules/", "*.log",
             "alias ll='ls -la'", "# end of file", "debug=true", "first line", "see you later", ".env", "foo", "bar baz",
             "set -e", "PATH=$PATH:~/bin", "this is a test", "remember the milk", "build/", "v1.0.0 released"]
TAGS = ["v1.0", "v1.2.0", "v2.0.0-rc1", "release-2024", "stable", "v0.3", "beta", "1.0.0", "prod-2024-05", "nightly"]
ORDINALS = ["second", "third", "2nd", "3rd", "fourth", "4th", "fifth", "5th", "first", "1st"]
RAW_COMMANDS = ["ls -la", "git status", "docker ps", "cat /etc/hosts", "make", "npm test", "df -h", "python script.py",
                "./build.sh", "grep -r foo .", "ps aux", "pacman -Syu", "systemctl restart nginx", "ls", "cargo build",
                "kubectl get pods", "tail -n 50 /var/log/syslog", "ip a", "du -sh *", "find . -name '*.py'", "mount /dev/sdb1 /mnt",
                "dmesg", "journalctl -xe", "apt update", "nmap -sn 192.168.1.0/24", "rm -rf build", "cat notes.txt",
                "python3 manage.py migrate", "git log --oneline", "free -h", "lsblk", "curl -s https://example.com", "top -bn1",
                "cp a.txt b.txt", "mkdir -p out/logs", "ss -tulpn", "env", "history", "uname -a", "wc -l *.txt", "pytest -q",
                "docker compose up -d", "chmod 600 key.pem", "vim config.yaml", "tar -xf data.tar.gz", "echo $PATH"]


def val(q, kind):
    """A random value of the given kind, as the user might type it -> (text, quote it?, words before, words after)."""
    r = q.r
    pick = r.choice
    if kind == "path":
        text, pre, post = q.folder()
        return text, False, pre, post
    if kind == "file":
        return q.filename(), False, "", ""
    if kind == "files":
        names = [q.filename() for _ in range(pick([1, 1, 1, 2, 2, 3]))]
        return (names[0] if len(names) == 1 else pkg_list(r, names)), False, "", ""
    if kind == "word":
        return q.word() + pick(["", "", "", "-fix", "2", "_x"]), False, "", ""
    if kind == "words":
        return (pkg_list(r, r.sample(PKGS, pick([1, 1, 1, 1, 2, 2, 3, 4]))) if r.random() < 0.8 else q.word()), False, "", ""
    if kind == "pkg":
        return (pick(PKGS) if r.random() < 0.75 else q.word()), False, "", ""
    if kind == "text":
        t = pick(PHRASES) if r.random() < 0.6 else q.word()
        return t, " " in t or r.random() < 0.2, "", ""
    if kind == "sentence":
        t = pick(SENTENCES + PHRASES[:10])
        return t, " " in t or r.random() < 0.6, "", ""
    if kind == "msg":
        t = pick(MSGS) if r.random() < 0.7 else " ".join(r.sample(WORDS, pick([1, 2, 3])))
        return t, r.random() < 0.55, "", ""
    if kind == "count":
        return q.count(), False, "", ""
    if kind == "smallcount":
        return str(pick([1, 2, 2, 3, 3, 4, 5])) if r.random() < 0.8 else pick(["one", "two", "three", "four"]), False, "", ""
    if kind == "size":
        return q.size(), False, "", ""
    if kind == "dur":
        return q.duration(), False, "", ""
    if kind == "ext":
        e = pick(EXT_NAMES) if r.random() < 0.3 else pick(EXTS)
        return (pick([e, e, "." + e]) if e in EXTS else e), False, "", ""
    if kind == "unit":
        u = pick(UNITS) if r.random() < 0.75 else q.word() + pick(["", "d", "-agent", "-server"])
        return u + (".service" if r.random() < 0.06 else ""), False, "", ""
    if kind == "host":
        return (pick(HOSTS) if r.random() < 0.8 else q.word() + pick([".com", ".local", "-01", ".lan", ".org", ""])), False, "", ""
    if kind == "url":
        if r.random() < 0.75:
            return pick(URLS), False, "", ""
        return f"https://{q.word()}.{pick(['com', 'org', 'io', 'net', 'dev'])}/{q.word()}{pick(['', '.zip', '.git', '.tar.gz', '/latest', '.json', '.pdf'])}", False, "", ""
    if kind == "port":
        return (pick(PORTS) if r.random() < 0.75 else str(r.randint(1024, 65000))), False, "", ""
    if kind == "pct":
        return (pick(PCTS) if r.random() < 0.7 else str(r.randint(1, 100)) + pick(["", "%", " percent"])), False, "", ""
    if kind == "envname":
        return (pick(ENVNAMES) if r.random() < 0.75 else (q.word() + pick(["", "_" + q.word()])).upper()), False, "", ""
    if kind == "cmdname":
        return (pick(CMDNAMES) if r.random() < 0.7 else pick(PKGS)), False, "", ""
    if kind == "container":
        return (pick(CONTAINERS) if r.random() < 0.75 else q.word() + pick(["", "-1", "_app", "-db"])), False, "", ""
    if kind == "image":
        return (pick(IMAGES) if r.random() < 0.75 else q.word() + pick(["", ":latest", ":1.0", ":dev"])), False, "", ""
    if kind == "script":
        return (pick(SCRIPTS) if r.random() < 0.8 else q.word()), False, "", ""
    if kind == "ssid":
        t = pick(SSIDS) if r.random() < 0.7 else q.word().capitalize() + pick(["", "-5G", " Guest", "WiFi", "_2.4"])
        return t, " " in t, "", ""
    if kind == "branch":
        return (pick(BRANCHES) if r.random() < 0.7 else q.word() + pick(["", "-fix", "-wip", "/v2"])), False, "", ""
    if kind == "expr":
        return pick(EXPRS) if r.random() < 0.7 else f"{r.randint(2, 999)} {pick(['*', '+', '-', '/', 'times', 'plus', 'minus'])} {r.randint(2, 99)}", False, "", ""
    if kind == "date":
        return pick(DATES), False, "", ""
    if kind == "owner":
        return (pick(OWNERS) if r.random() < 0.8 else q.word()), False, pick(["", "", "", "user", "the user"]), ""
    if kind == "mode":
        return pick(MODE_WORDS), False, "", ""
    if kind == "remote":
        return (pick(REMOTES) if r.random() < 0.8 else f"{q.word()}@{q.word()}:{pick(['', '~/', '/srv/', '/tmp/x'])}"), False, "", ""
    if kind == "affix":
        return pick(AFFIXES), r.random() < 0.15, "", ""
    if kind == "tag":
        return pick(TAGS), False, "", ""
    if kind == "alias":
        return pick(ALIASES), False, "", ""
    if kind == "aliascmd":
        t = pick(ALIAS_CMDS)
        return t, " " in t or r.random() < 0.5, "", ""
    if kind == "varvalue":
        return pick(VAR_VALUES), r.random() < 0.15, "", ""
    if kind == "script_file":
        f = q.filename(["sh", "py", "js", "rb", "pl", "run", "bin"])
        return (("./" + f.split("/")[-1]) if r.random() < 0.3 else f), False, "", ""
    if kind == "archive":
        return q.filename(["zip", "tar.gz", "tgz", "tar", "7z", "rar", "tar.xz", "gz", "tar.bz2"]), False, "", ""
    if kind == "textfile":
        return q.filename(["log", "txt", "md", "conf", "json", "py", "csv", "yaml", "ini", "sh", "toml", "cfg", "env"]), False, "", ""
    if kind == "target":
        roll = r.random()
        if roll < 0.55:
            return q.filename(["pdf", "png", "jpg", "html", "mp4", "docx", "txt", "svg", "xlsx", "mkv", "odt"]), False, "", ""
        if roll < 0.75:
            return pick(["firefox", "chromium", "gimp", "vlc", "code", "dolphin", "steam", "libreoffice", "thunderbird", "spotify", "discord", "kate", "konsole", "obs"]), False, "", ""
        if roll < 0.87:
            return pick(URLS), False, "", ""
        text, pre, post = q.folder()
        return text, False, pre, post
    if kind == "bindir":
        return pick(["~/bin", "~/.local/bin", "/opt/tool/bin", "~/go/bin", "~/.cargo/bin", "/usr/local/go/bin", "./node_modules/.bin", "~/scripts", "/opt/" + q.word() + "/bin"]), False, "", ""
    if kind == "venv":
        return pick([".venv", "venv", "env", "~/envs/" + q.word(), ".env", "myenv", "~/.virtualenvs/" + q.word()]), False, "", ""
    if kind == "binfile":
        return pick(["/usr/bin/" + pick(CMDNAMES), "/etc/" + q.word() + ".conf", pick(CMDNAMES), "/usr/lib/lib" + q.word() + ".so", "/usr/bin/" + q.word()]), False, "", ""
    raise ValueError(kind)


def put(q, kind):
    """Emit a value of this kind into the request and return its key."""
    text, quote, pre, post = val(q, kind)
    q.lit(pre)
    k = q.key()
    q.chunks.append((f'"{text}"' if quote else text, k))
    q.lit(post)
    return k



class Req:
    """One request under construction: English chunks plus the program that goes with them."""

    def __init__(self, r):
        self.r = r
        self.chunks = []          # (text, key or None)
        self.n = 0

    def key(self):
        self.n += 1
        return f"k{self.n}"

    def pick(self, options):
        return self.r.choice(options)

    def chance(self, p):
        return self.r.random() < p

    # ---- values
    def word(self):
        return self.pick(WORDS)

    def count(self):
        if self.chance(0.3):
            return self.pick(NUM_WORDS)
        return str(self.pick([1, 2, 3, 3, 5, 5, 10, 10, 10, 15, 20, 20, 25, 30, 50, 100, self.r.randint(2, 99)]))

    def size(self):
        n = self.pick([1, 2, 5, 10, 20, 50, 100, 200, 250, 500, 700, 1.5, 2.5, self.r.randint(1, 999)])
        style = self.r.random()
        if style < 0.45:
            return f"{n}{self.pick(['MB', 'GB', 'KB', 'mb', 'gb', 'kb', 'M', 'G', 'k', 'K', 'm', 'g', 'MiB', 'GiB'])}"
        if style < 0.9:
            unit = self.pick(["mb", "gb", "kb", "MB", "GB", "megabytes", "gigabytes", "megs", "gigs", "kilobytes", "megabyte", "gigabyte", "gig", "meg"])
            return f"{n} {unit}"
        return self.pick(["a gig", "a gigabyte", "half a gig", "a megabyte", "one gb", "a meg", "one megabyte", "two gigs"])

    def duration(self):
        style = self.r.random()
        n = self.pick([1, 2, 3, 5, 7, 10, 14, 30, 60, 90, self.r.randint(2, 48)])
        if style < 0.55:
            unit = self.pick(["days", "weeks", "months", "hours", "minutes", "years", "days", "days", "weeks", "hours"])
            if n == 1:
                unit = unit[:-1]
            return f"{self.pick(NUM_WORDS) if self.chance(0.2) else n} {unit}"
        if style < 0.75:
            return self.pick(["a day", "a week", "a month", "a year", "an hour", "one week", "one month", "a couple days",
                              "a few days", "a couple of weeks", "half an hour", "a fortnight"])
        return f"{n}{self.pick(['d', 'h', 'w', 'min', 'm', 'y', 'mo', 'days', 'hrs', 'wks'])}"

    def filename(self, exts=None):
        if self.chance(0.17):
            return self.pick(SPECIAL_FILES)
        base = self.word()
        if self.chance(0.4):
            base += self.pick(["_", "-", ""]) + self.pick(WORDS + ["2023", "2024", "v2", "01", "final", "old"])
        name = f"{base}.{self.pick(exts or EXTS)}"
        if self.chance(0.25):
            name = f"{self.pick(REAL_DIRS)}/{name}"
        return name

    def folder(self):
        """A folder as the user would say it: (surface text for the span, extra words before, extra words after)."""
        style = self.r.random()
        if style < 0.34:
            return self.pick(REAL_DIRS), "", ""
        if style < 0.46:
            return f"{self.pick(REAL_DIRS)}/{self.pick(WORDS + DIR_WORDS)}", "", ""
        if style < 0.62:
            return self.pick(DIR_WORDS + WORDS[:60]), "", ""
        if style < 0.85:
            return self.pick(FOLDER_NAMES), self.pick(["the", "my", "the", ""]), self.pick(["folder", "directory", "", "dir", "folder"])
        return self.word(), self.pick(["the", "my", "a", ""]), self.pick(["folder", "directory", "dir"])

    # ---- building
    def lit(self, text):
        if text:
            self.chunks.append((text, None))

    def arg(self, text, quote=False):
        k = self.key()
        self.chunks.append((f'"{text}"' if quote else text, k))
        return k

    def folder_arg(self, prep_options):
        text, pre, post = self.folder()
        self.lit(self.pick(prep_options))
        self.lit(pre)
        k = self.arg(text)
        self.lit(post)
        return k


def finish(req, program, ctx=None):
    """Resolve argument keys to word positions; return (words, output token ids, number of request words).
    A slot value ("ctx", i) points at word i of the context, which is placed after the request's own words."""
    words, spans = [], {}
    for text, key in req.chunks:
        ws = tok.words(text)
        if key is not None:
            spans[key] = (len(words), len(words) + len(ws) - 1)
        words += ws
    n = len(words)
    resolved = []
    for pipe in program:
        rp = []
        for name, args in pipe:
            ra = {}
            for slot in ir.BY_NAME[name].slots:
                v = args.get(slot.name)
                if slot.kind == "span" and v is not None:
                    v = (n + v[1], n + v[1]) if isinstance(v, tuple) else spans[v]
                ra[slot.name] = v
            rp.append((name, ra))
        resolved.append(rp)
    return words + list(ctx or []), ir.linearise(resolved), n


def filter_stages(stages):
    """Filters in the fixed order the language uses, each negated one preceded by "not"."""
    out = []
    for n in ir.FILTER_ORDER:
        if n in stages:
            if n in stages.get("__not__", ()):
                out.append(("not", {}))
            out.append((n, stages[n]))
    return out


# --------------------------------------------------------------------------- file requests
MASS = {"stuff", "things", "anything", "everything", "content", "data"}
NEUTRAL = ["random", "stray", "leftover", "unused", "junk", "useless", "extra", "various", "misc", "loose", "spare", "assorted"]
VAGUE = {"larger": ["large", "big", "huge", "giant", "massive", "heavy", "bulky", "fat", "enormous", "really big", "very large"],
         "smaller": ["small", "tiny", "little", "really small", "very small", "teeny"],
         "older": ["old", "ancient", "stale", "outdated", "really old", "very old", "aging", "dusty"],
         "newer": ["new", "recent", "fresh", "recently changed", "recently modified", "brand new"]}
SUPER = {
    ("size", "desc"): ["largest", "biggest", "heaviest", "largest", "biggest", "bulkiest", "fattest"],
    ("size", "asc"): ["smallest", "tiniest", "lightest", "smallest"],
    ("time", "desc"): ["newest", "most recent", "latest", "most recently modified", "most recently changed",
                       "most recently edited", "freshest", "most recently updated"],
    ("time", "asc"): ["oldest", "least recently modified", "least recently used", "earliest", "oldest"],
}
SORT_CLAUSE = {
    ("size", "desc"): ["sorted by size", "biggest first", "largest first", "ordered by size", "from largest to smallest",
                       "by size", "in order of size", "sorted by size descending", "with the biggest at the top"],
    ("size", "asc"): ["smallest first", "sorted by size ascending", "from smallest to largest", "ordered by size smallest first"],
    ("time", "desc"): ["newest first", "sorted by date", "most recent first", "ordered by modification time",
                       "by date", "sorted by time", "latest first", "sorted by when they were changed"],
    ("time", "asc"): ["oldest first", "sorted by date oldest first", "from oldest to newest", "earliest first"],
    ("name", "asc"): ["sorted by name", "alphabetically", "in alphabetical order", "ordered by name", "sorted a to z"],
    ("name", "desc"): ["in reverse alphabetical order", "sorted by name descending", "sorted z to a"],
}
SHOW = ["show", "show me", "list", "find", "display", "print", "give me", "get", "look for", "search for", "locate",
        "find me", "list out", "i want to see", "let me see", "pull up", "find all", "list all", "show all",
        "what are", "which are", "where are", "get me", "dig up", "track down", "ls", "tell me", "look up", "see",
        "show me all", "can i see", "fetch", "get a list of", "give me a list of", "find any", "i need", "i need to see",
        "point me to", "bring up", "check for", "scan for", "hunt for"]
DELETE = ["delete", "remove", "get rid of", "erase", "wipe", "clean up", "purge", "rm", "nuke", "clear out",
          "throw away", "delete all", "remove all", "wipe out", "destroy", "discard", "drop", "clean out", "kill off"]
MOVE = ["move", "transfer", "relocate", "mv", "shift", "put", "move all", "send", "dump"]
COPY = ["copy", "duplicate", "back up", "cp", "copy all", "make a copy of", "clone", "save a copy of", "backup"]
TGZ = ["compress", "archive", "tar up", "make a tarball of", "pack", "bundle", "tar", "make an archive of", "gzip", "pack up"]
ZIP = ["zip", "zip up", "make a zip of", "create a zip of", "zip together", "put in a zip"]


def file_filters(q, dirs):
    """Sample filters. Returns (canonical stages, phrases before the noun, phrases after the noun, ext-as-noun or None).
    stages["__not__"] holds the names of negated filters."""
    r = q.r
    stages, pre, post, noun, negated = {}, [], [], None, set()
    names = ["ext", "name", "larger", "smaller", "newer", "older", "contains", "empty", "since", "before", "hidden", "owner", "isexec", "depth"]
    probs = ([0.55, 0.14, 0.2, 0.06, 0.2, 0.2, 0.12, 0.05, 0.035, 0.03, 0.03, 0.03, 0.025, 0.02] if not dirs
             else [0, 0.3, 0, 0, 0.2, 0.2, 0, 0.2, 0.03, 0.02, 0.04, 0.04, 0, 0.03])
    chosen = [n for n, p in zip(names, probs) if r.random() < p]
    for a, b in (("newer", "older"), ("since", "newer"), ("before", "older"), ("since", "before")):
        if a in chosen and b in chosen:
            chosen.remove(r.choice([a, b]))
    if "larger" in chosen and "smaller" in chosen and r.random() < 0.7:
        chosen.remove("smaller")
    if "empty" in chosen:
        chosen = [c for c in chosen if c not in ("larger", "smaller", "contains")]
    r.shuffle(chosen)
    chosen = chosen[:r.choice([0, 1, 1, 1, 2, 2, 2, 3, 3, 4])]
    for n in chosen:
        neg = n in ("ext", "name", "contains", "empty", "hidden", "owner", "isexec") and r.random() < 0.07
        if neg:
            negated.add(n)
        if n == "ext":
            if r.random() < 0.07:                       # several types at once
                es = r.sample(EXTS, r.choice([2, 2, 3]))
                e = (" " + r.choice(["or", "and"]) + " ").join(es) if len(es) == 2 else f"{es[0]}, {es[1]} {r.choice(['or', 'and'])} {es[2]}"
            else:
                e = None
            k = q.key()
            if neg:
                e = e or (r.choice(EXT_NAMES) if r.random() < 0.4 else r.choice(EXTS + EXT_PLURALS))
                lead = r.choice(["that are not", "except", "other than", "excluding", "but not", "that aren't", "ignoring",
                                 "apart from", "not including", "except for", "but skip", "leaving out", "except the"])
                post.append([lead, (e, k), r.choice(["files", "", "", "ones"]) if e not in EXT_PLURALS else ""])
            elif e is not None:
                post.append([r.choice(["that are", "of type", "ending in", "with extension", "in"]), (e, k), r.choice(["", "files", "format"])]) if r.random() < 0.4 else pre.append([(e, k)])
            else:
                style = r.random()
                if style < 0.14:
                    noun = (r.choice(EXT_PLURALS), k)
                elif style < 0.6:
                    e = r.choice(EXT_NAMES) if r.random() < 0.45 else r.choice(EXTS)
                    e = r.choice([e, e, "." + e, "*." + e]) if e in EXTS else e
                    pre.append([(e, k)])
                else:
                    e = r.choice(EXTS)
                    lead = r.choice(["ending in", "with the extension", "with extension", "of type", "with the", "that end in",
                                     "whose extension is", "that are", "in", "with a", "having the extension"])
                    e2 = ("." + e) if lead in ("ending in", "that end in") or r.random() < 0.5 else e
                    tail = {"with the": "extension", "with a": "extension", "in": "format", "that are": r.choice(["files", ""])}.get(lead, "")
                    post.append([lead, (e2, k), tail])
            stages["ext"] = {"ext": k}
        elif n == "name":
            mode = r.choice(["is", "has", "has", "starts", "starts", "ends"])
            if mode == "is":
                t = q.filename() if not dirs else q.word()
                if r.random() < 0.25:
                    t = r.choice(["*", ""]) + q.word() + "*"
                lead = r.choice(["named", "called", "with the name", "whose name is", "by the name of", "matching", "named exactly"])
                nlead = r.choice(["not named", "not called", "whose name is not", "that aren't called", "except the one named"])
            elif mode == "has":
                t = r.choice(WORDS + ["2023", "2024", "v2", "old", "copy", "final"])
                lead = None
                nlead = r.choice(["whose name doesn't contain", "without", "with names not containing", "that don't have", "not containing"])
            elif mode == "starts":
                t = r.choice(WORDS + ["tmp", "IMG", "DSC", "test_", "draft-", "2024", "."])
                lead = r.choice(["starting with", "that start with", "whose name begins with", "beginning with",
                                 "whose names start with", "with names starting with", "prefixed with", "that begin with"])
                nlead = r.choice(["not starting with", "that don't start with", "whose name doesn't begin with", "that do not begin with"])
            else:
                t = r.choice(["_old", "-backup", "copy", "2023", "_final", "~", "-v2", "_test", "bak", ".orig"])
                lead = r.choice(["ending with", "whose name ends in", "that end with", "whose names end with",
                                 "with names ending in", "suffixed with"])
                nlead = r.choice(["not ending with", "that don't end with", "whose name doesn't end in"])
            k = q.key()
            if neg:
                post.append([nlead, (t, k)] + (["in the name"] if mode == "has" and nlead in ("without", "that don't have", "not containing") else []))
            elif dirs and mode == "is" and r.random() < 0.45:
                pre.append([(r.choice(["node_modules", "__pycache__", ".git", "build", "dist", "target", "venv", ".cache", "tmp"] + WORDS[:40]), k)])
            elif lead is None:
                form = r.randrange(6)
                post.append([["with", (t, k), "in the name"], ["whose name contains", (t, k)], ["with names containing", (t, k)],
                             ["that have", (t, k), "in their name"], ["with", (t, k), "in their names"],
                             ["containing", (t, k), "in the filename"]][form])
            else:
                post.append([lead, (t, k)])
            stages["name"] = {"mode": mode, "text": k}
        elif n in ("larger", "smaller"):
            if r.random() < 0.14:                       # "large files": no number given
                adj = r.choice(VAGUE[n])
                if r.random() < 0.7:
                    pre.append([adj])
                else:
                    post.append([r.choice(["that are", "which are", "that look", "that seem"]), adj])
                stages[n] = {"size": None}
                continue
            k = q.key(); s = q.size()
            lead = r.choice(["larger than", "bigger than", "over", "above", "more than", "greater than", "that are bigger than",
                             "exceeding", "that are larger than", "which are over", "that take more than", "at least",
                             "heavier than", "that exceed", "weighing more than", "bigger then", "larger then"]
                            if n == "larger" else
                            ["smaller than", "under", "below", "less than", "that are smaller than", "which are under",
                             "at most", "lighter than", "that are less than", "not bigger than", "smaller then"])
            post.append([lead, (s, k)])
            stages[n] = {"size": k}
        elif n == "newer":
            if r.random() < 0.12:
                adj = r.choice(VAGUE[n])
                if r.random() < 0.7:
                    pre.append([adj])
                else:
                    post.append([r.choice(["that are", "which are"]), adj])
                stages[n] = {"age": None}
                continue
            k = q.key()
            if r.random() < 0.3:
                d = r.choice(["today", "yesterday", "this week", "this month", "this year", "last week", "last month"])
                lead = r.choice(["modified", "changed", "from", "edited", "updated", "touched", "created", "that changed",
                                 "that were modified", "i changed", "that i edited", "modified since", "changed since", "since"])
                post.append([lead, (d, k)])
            else:
                d = q.duration()
                form = r.randrange(12)
                post.append([["modified in the last", (d, k)], ["changed in the past", (d, k)], ["from the last", (d, k)],
                             ["newer than", (d, k)], ["edited within the last", (d, k)], ["that changed in the last", (d, k)],
                             ["less than", (d, k), "old"], ["created in the last", (d, k)], ["updated in the last", (d, k)],
                             ["modified within", (d, k)], ["touched in the last", (d, k)], ["from the past", (d, k)]][form])
            stages["newer"] = {"age": k}
        elif n == "older":
            if r.random() < 0.12:
                adj = r.choice(VAGUE[n])
                if r.random() < 0.7:
                    pre.append([adj])
                else:
                    post.append([r.choice(["that are", "which are"]), adj])
                stages[n] = {"age": None}
                continue
            k = q.key(); d = q.duration()
            form = r.randrange(13)
            post.append([["older than", (d, k)], ["not modified in", (d, k)], ["that haven't been touched in", (d, k)],
                         ["that haven't changed in over", (d, k)], ["more than", (d, k), "old"],
                         ["last modified more than", (d, k), "ago"], ["untouched for", (d, k)], ["from more than", (d, k), "ago"],
                         ["at least", (d, k), "old"], ["that are older than", (d, k)], ["not changed for", (d, k)],
                         ["unchanged for over", (d, k)], ["that nobody has touched in", (d, k)]][form])
            stages["older"] = {"age": k}
        elif n == "since":
            k = q.key(); d = r.choice(DATES)
            post.append([r.choice(["modified since", "changed after", "from", "modified after", "created after", "since",
                                   "edited after", "that changed on or after", "updated since", "newer than", "touched after",
                                   "changed since", "modified on or after", "dated after"]), (d, k)]
                        + (["onwards"] if r.random() < 0.1 else []))
            stages["since"] = {"date": k}
        elif n == "before":
            k = q.key(); d = r.choice(DATES)
            post.append([r.choice(["modified before", "from before", "last changed before", "that haven't changed since",
                                   "created before", "edited before", "older than", "untouched since", "dated before",
                                   "changed before", "not modified since"]), (d, k)])
            stages["before"] = {"date": k}
        elif n == "contains":
            k = q.key()
            t = r.choice(PHRASES) if r.random() < 0.6 else q.word()
            quote = " " in t or r.random() < 0.2
            t2 = (t, k, quote)
            if neg:
                form = r.randrange(7)
                post.append([["that don't contain", t2], ["that do not mention", t2], ["without", t2, "in them"], ["not containing", t2],
                             ["that lack", t2], ["missing the text", t2], ["where", t2, "does not appear"]][form])
            else:
                form = r.randrange(11)
                post.append([["containing", t2], ["that contain", t2], ["that mention", t2], ["with", t2, "in them"],
                             ["which include the text", t2], ["that have the string", t2], ["mentioning", t2],
                             ["with the word", t2], ["where", t2, "appears"], ["that include", t2], ["with", t2, "inside"]][form])
            stages["contains"] = {"text": k}
        elif n == "empty":
            if neg:
                if r.random() < 0.4:
                    pre.append([r.choice(["non-empty", "nonempty", "non empty"])])
                else:
                    post.append([r.choice(["that are not empty", "that aren't empty", "which have something in them",
                                           "that are not blank", "with something in them"])])
            elif r.random() < 0.5:
                pre.append(["empty"])
            else:
                post.append([r.choice(["that are empty", "with nothing in them", "which are empty", "that have nothing in them",
                                       "of zero size" if not dirs else "with no files inside"])])
            stages["empty"] = {}
        elif n == "hidden":
            if neg:
                post.append([r.choice(["that are not hidden", "excluding hidden ones", "ignoring dotfiles", "but not hidden ones",
                                       "skipping hidden ones", "except hidden ones", "that don't start with a dot"])])
            elif r.random() < 0.6:
                pre.append(["hidden"])
            else:
                post.append([r.choice(["that are hidden", "starting with a dot", "which are hidden", "whose names begin with a dot"])])
            stages["hidden"] = {}
        elif n == "owner":
            k = q.key()
            o = r.choice(OWNERS)
            if o in ("me", "myself") and not neg and r.random() < 0.5:
                post.append([r.choice(["that", "which"]), ("i", k), r.choice(["own", "am the owner of"])])
            elif neg:
                post.append([r.choice(["not owned by", "that don't belong to", "not belonging to", "that aren't owned by", "owned by someone other than"]), (o, k)])
            else:
                post.append([r.choice(["owned by", "belonging to", "that belong to", "whose owner is", "that are owned by", "of user", "created by"]), (o, k)])
            stages["owner"] = {"user": k}
        elif n == "isexec":
            if neg:
                post.append([r.choice(["that are not executable", "that aren't executable", "without the execute bit", "that can't be run"])])
            elif r.random() < 0.6:
                pre.append(["executable"])
            else:
                post.append([r.choice(["that are executable", "i can run", "with the execute bit", "that can be run", "marked executable"])])
            stages["isexec"] = {}
        elif n == "depth":
            k = q.key(); d = r.choice(["1", "2", "2", "3", "3", "4", "5", "two", "three"])
            form = r.randrange(6)
            post.append([["at most", (d, k), "levels deep"], ["up to", (d, k), "levels down"], ["no deeper than", (d, k), "levels"],
                         ["within", (d, k), "levels"], ["max depth", (d, k)], ["going", (d, k), "levels deep"]][form])
            stages["depth"] = {"n": k}
    if negated:
        stages["__not__"] = negated
    return stages, pre, post, noun


CONNECT = ["and", ",", "but", "and also", "and", "that are also", "plus", "and are", ", and"]


def tail(q, bits):
    """Emit the clauses that follow the noun, sometimes joined with 'and', 'but', a comma..."""
    for i, t in enumerate(bits):
        if i and q.r.random() < 0.3:
            q.lit(q.r.choice(CONNECT))
        emit(q, t)


def emit(q, phrase):
    for part in phrase:
        if isinstance(part, str):
            q.lit(part)
        else:
            text, key = part[0], part[1]
            quote = part[2] if len(part) > 2 else False
            q.chunks.append((f'"{text}"' if quote else text, key))



class A(str):
    """A slot of the action, inside a frame."""


FRAMES = {
    "trash": [["trash", "NP"], ["move", "NP", "to the trash"], ["send", "NP", "to the trash"], ["recycle", "NP"], ["bin", "NP"],
              ["put", "NP", "in the trash"], ["move", "NP", "to the recycle bin"], ["throw", "NP", "in the bin"], ["move", "NP", "to trash"]],
    "details": [["show", "NP", "with details"], ["long listing of", "NP"], ["ls -l", "NP"], ["list", "NP", "with permissions and sizes"],
                ["show", "NP", "with their permissions"], ["show", "NP", "with owner and date"], ["detailed list of", "NP"],
                ["list", "NP", "in long format"], ["show permissions of", "NP"], ["who owns", "NP"], ["list", "NP", "with full details"]],
    "fgrep": [["search", "NP", "for", A("text")], ["grep for", A("text"), "in", "NP"], ["find", A("text"), "in", "NP"],
              ["look for", A("text"), "in", "NP"], ["search for", A("text"), "in", "NP"], ["grep", A("text"), "in", "NP"],
              ["show lines containing", A("text"), "in", "NP"], ["where is", A("text"), "used in", "NP"],
              ["find every line mentioning", A("text"), "in", "NP"], ["grep -r", A("text"), "NP"],
              ["which lines in", "NP", "mention", A("text")], ["find where", A("text"), "appears in", "NP"],
              ["show me every place", A("text"), "shows up in", "NP"], ["search through", "NP", "for", A("text")]],
    "fopen": [["open", "NP"], ["open up", "NP"], ["xdg-open", "NP"], ["open", "NP", "with the default app"]],
    "fedit": [["edit", "NP"], ["open", "NP", "in my editor"], ["open", "NP", "in vim"], ["open", "NP", "for editing"], ["let me edit", "NP"]],
    "chmod": [["make", "NP", A("mode")], ["chmod", A("mode"), "NP"], ["set the permissions of", "NP", "to", A("mode")],
              ["give", "NP", A("mode"), "permissions"], ["change the mode of", "NP", "to", A("mode")], ["set", "NP", "to", A("mode")],
              ["mark", "NP", "as", A("mode")], ["change permissions on", "NP", "to", A("mode")], ["make", "NP", "all", A("mode")]],
    "chown": [["change the owner of", "NP", "to", A("owner")], ["chown", A("owner"), "NP"], ["give", "NP", "to", A("owner")],
              ["make", A("owner"), "the owner of", "NP"], ["set the owner of", "NP", "to", A("owner")],
              ["change ownership of", "NP", "to", A("owner")], ["transfer ownership of", "NP", "to", A("owner")]],
    "rn_lower": [["rename", "NP", "to lowercase"], ["lowercase the names of", "NP"], ["make the names of", "NP", "lowercase"],
                 ["convert the filenames of", "NP", "to lower case"], ["make", "NP", "have lowercase names"],
                 ["lowercase", "NP", "filenames"], ["rename", "NP", "so the names are all lowercase"], ["make the filenames of", "NP", "all lower case"],
                 ["make every filename", "LOC", "lowercase"], ["lowercase all the filenames", "LOC"], ["make all file names", "LOC", "lower case"]],
    "rn_upper": [["rename", "NP", "to uppercase"], ["uppercase the names of", "NP"], ["make the names of", "NP", "uppercase"],
                 ["convert the filenames of", "NP", "to upper case"], ["rename", "NP", "in all caps"], ["capitalize the names of", "NP"],
                 ["make the filenames of", "NP", "all caps"], ["make every filename", "LOC", "uppercase"], ["uppercase all the filenames", "LOC"]],
    "rn_spaces": [["replace spaces with underscores in the names of", "NP"], ["remove spaces from the names of", "NP"],
                  ["rename", "NP", "replacing spaces with underscores"], ["get rid of spaces in", "NP", "filenames"],
                  ["turn spaces into underscores in the names of", "NP"], ["fix the spaces in the names of", "NP"],
                  ["rename", "NP", "so there are no spaces in the names"], ["get rid of the spaces in every filename", "LOC"],
                  ["replace spaces with underscores in all filenames", "LOC"]],
    "rn_replace": [["rename", "NP", "replacing", A("old"), "with", A("new")], ["replace", A("old"), "with", A("new"), "in the names of", "NP"],
                   ["in the filenames of", "NP", "change", A("old"), "to", A("new")], ["rename", "NP", "changing", A("old"), "to", A("new")],
                   ["swap", A("old"), "for", A("new"), "in the names of", "NP"], ["rename", "NP", "so", A("old"), "becomes", A("new")],
                   ["change", A("old"), "to", A("new"), "in the filenames of", "NP"]],
    "rn_prefix": [["add the prefix", A("text"), "to", "NP"], ["prefix", "NP", "with", A("text")], ["prepend", A("text"), "to the names of", "NP"],
                  ["rename", "NP", "adding", A("text"), "at the start"], ["put", A("text"), "in front of the names of", "NP"],
                  ["add", A("text"), "to the beginning of the names of", "NP"]],
    "rn_suffix": [["add the suffix", A("text"), "to", "NP"], ["append", A("text"), "to the names of", "NP"],
                  ["rename", "NP", "adding", A("text"), "at the end"], ["suffix", "NP", "with", A("text")],
                  ["add", A("text"), "to the end of the names of", "NP"]],
    "rn_ext": [["change the extension of", "NP", "to", A("ext")], ["rename", "NP", "to", A("ext")], ["give", "NP", "the extension", A("ext")],
               ["change", "NP", "to", A("ext"), "files"], ["rename", "NP", "to have the", A("ext"), "extension"],
               ["switch the extension of", "NP", "to", A("ext")], ["change the file extension of", "NP", "to", A("ext")]],
    "replace": [["replace", A("old"), "with", A("new"), "in", "NP"], ["change every", A("old"), "to", A("new"), "in", "NP"],
                ["swap", A("old"), "for", A("new"), "in", "NP"], ["find and replace", A("old"), "with", A("new"), "in", "NP"],
                ["in", "NP", "replace", A("old"), "with", A("new")], ["substitute", A("new"), "for", A("old"), "in", "NP"],
                ["change", A("old"), "to", A("new"), "in", "NP"], ["sed", A("old"), "to", A("new"), "in", "NP"],
                ["replace all occurrences of", A("old"), "with", A("new"), "in", "NP"], ["replace every", A("old"), "with", A("new"), "across", "NP"]],
    "fhash": [["checksum", "NP"], ["sha256 of", "NP"], ["hash", "NP"], ["compute checksums for", "NP"], ["sha256sum", "NP"], ["get the hashes of", "NP"]],
    "by_ext": [["count", "NP", "by extension"], ["count", "NP", "by type"], ["how many", "NP1", "of each type"], ["break down", "NP", "by extension"],
               ["what kinds of", "NP1", "are there"], ["file types among", "NP"], ["group", "NP", "by extension and count"],
               ["how many of each file type", "LOC"], ["which file types are most common", "LOC"], ["breakdown of file types", "LOC"],
               ["what types of files are", "LOC"], ["count files per extension", "LOC"]],
    "dupes": [["find duplicate", "NP1"], ["which", "NP1", "are duplicates"], ["find duplicates among", "NP"], ["duplicate", "NP1"],
              ["find identical", "NP1"], ["are there duplicate", "NP1"], ["which", "NP1", "are identical"],
              ["find", "NP", "that are copies of each other"], ["look for duplicated", "NP1"], ["any duplicate", "NP1"]],
    "upload": [["upload", "NP", "to", A("to")], ["send", "NP", "to", A("to")], ["rsync", "NP", "to", A("to")],
               ["copy", "NP", "to the server", A("to")], ["scp", "NP", "to", A("to")], ["sync", "NP", "to", A("to")],
               ["push", "NP", "to", A("to")], ["transfer", "NP", "to", A("to")], ["copy", "NP", "over to", A("to")]],
    "names_save": [["save the list of", "NP", "to", A("to")], ["write the names of", "NP", "to", A("to")], ["list", "NP", "into", A("to")],
                   ["put the paths of", "NP", "in", A("to")], ["save a list of", "NP", "in", A("to")],
                   ["export the names of", "NP", "to", A("to")], ["dump the list of", "NP", "into", A("to")]],
    "cat_count": [["count the lines in", "NP"], ["how many lines of code in", "NP"], ["how many lines are in", "NP"],
                  ["total lines across", "NP"], ["count lines of", "NP"], ["how many lines do", "NP", "have in total"],
                  ["line count of", "NP"], ["lines of code in", "NP"], ["wc -l on", "NP"], ["how many lines of code are in", "NP"]],
    "cat_wcount": [["how many words are in", "NP"], ["word count of", "NP"], ["count the words in", "NP"], ["total words across", "NP"]],
}
FRAME_ARGS = {"fgrep": {"text": "text"}, "chmod": {"mode": "mode"}, "chown": {"owner": "owner"},
              "rn_replace": {"old": "word", "new": "word"}, "rn_prefix": {"text": "affix"}, "rn_suffix": {"text": "affix"},
              "rn_ext": {"ext": "ext"}, "replace": {"old": "text", "new": "text"}, "upload": {"to": "remote"},
              "names_save": {"to": "textfile"}}
FRAME_WT = {"fgrep": 7, "trash": 3, "replace": 3, "details": 2.5, "chmod": 2.5, "rn_lower": 1.5, "rn_replace": 2, "rn_ext": 1.5,
            "upload": 2.5, "cat_count": 3, "by_ext": 1.5, "dupes": 1.5, "fopen": 1.5, "names_save": 1.5}
FRAME_DIRS = {"trash", "details", "chmod", "chown", "rn_lower", "rn_upper", "rn_spaces", "rn_replace", "rn_prefix", "rn_suffix",
              "upload", "fopen", "names_save"}
FRAME_SUPER = {"trash", "fopen", "fedit", "upload", "details", "fhash", "chmod"}


def frame_emit(q, sink, np, tailbits, loc_done=False):
    """Write the request for one of the frame actions; returns the action's slot keys."""
    r = q.r
    seq = r.choice(FRAMES[sink])
    if "LOC" in seq and any(not isinstance(part, str) for part in np):
        seq = r.choice([f for f in FRAMES[sink] if "NP" in f or "NP1" in f])      # the noun phrase carries a filter: it must be said
    args = {}
    used_np = False
    for part in seq:
        if isinstance(part, A):
            text, quote, pre, post = val(q, FRAME_ARGS[sink][part])
            q.lit(pre)
            args[str(part)] = q.key()
            q.chunks.append((f'"{text}"' if quote else text, args[str(part)]))
            q.lit(post)
        elif part in ("NP", "NP1"):
            emit(q, np if part == "NP" else np[1:]); tail(q, tailbits); used_np = True
        elif part == "LOC":
            tail(q, tailbits); used_np = True
        else:
            q.lit(part)
    if not used_np:
        tail(q, tailbits)
    return args


def frame_stages(sink, args):
    if sink == "names_save":
        return [("fnames", {}), ("save", {"to": args["to"]})]
    if sink == "cat_count":
        return [("fcat", {}), ("lcount", {})]
    if sink == "cat_wcount":
        return [("fcat", {}), ("wcount", {})]
    return [(sink, args)]


def files_request(q):
    r = q.r
    dirs = r.random() < 0.12
    frames = [f for f in FRAMES if not dirs or f in FRAME_DIRS]
    sink = r.choices(["fshow", "delete", "fcount", "total", "sizes", "move", "copy", "compress"] + frames,
                     ([30, 14, 10, 6, 5, 8, 8, 8] if not dirs else [30, 10, 10, 2, 22, 5, 5, 4]) + [FRAME_WT.get(f, 1.0) for f in frames])[0]
    stages, pre, post, ext_noun = file_filters(q, dirs)
    # superlative / sort
    sort = take = pick = None
    sup_phrase, sort_clause = None, None
    singular = False
    if (sink in ("fshow", "delete", "move", "copy", "compress", "sizes") or sink in FRAME_SUPER) and r.random() < 0.3:
        key = r.choice([("size", "desc"), ("size", "desc"), ("size", "asc"), ("time", "desc"), ("time", "desc"), ("time", "asc")])
        sort = key
        word = r.choice(SUPER[key])
        style = r.random()
        if style < 0.07 and sink != "sizes":
            k = q.key(); pick = k; singular = True
            sup_phrase = ["the", (r.choice(ORDINALS), k), word]
        elif style < 0.6:
            n = q.count(); k = q.key(); take = k
            sup_phrase = r.choice([["the", (n, k), word], [(n, k), word], ["the", word, (n, k)], ["top", (n, k), word],
                                   ["the top", (n, k), word], ["the", (n, k), word], [word, (n, k)]])
        elif style < 0.8:
            singular = True; take = "none"
            sup_phrase = ["the", word] if r.random() < 0.8 else [word]
        else:
            sup_phrase = ["the", word] if r.random() < 0.7 else [word]
    elif sink == "fshow" and r.random() < 0.12:
        key = r.choice(list(SORT_CLAUSE))
        sort = key
        sort_clause = r.choice(SORT_CLAUSE[key])
    # location
    loc_key, loc_phrase, flat = None, None, False
    if r.random() < 0.68:
        text, lpre, lpost = q.folder()
        loc_key = q.key()
        if r.random() < 0.07:
            flat = True
            form = r.randrange(5)
            core = [lpre, (text, loc_key), lpost]
            loc_phrase = [["directly in"] + core, ["at the top level of"] + core, ["in"] + core + ["but not its subfolders"],
                          ["in"] + core + ["only, not subfolders"], ["just in"] + core + ["itself"]][form]
        else:
            prep = r.choice(["in", "in", "in", "under", "inside", "within", "from", "in", "inside of", "located in", "that are in", "sitting in"])
            if sink in ("move", "copy") and r.random() < 0.5:
                prep = r.choice(["from", "in", "out of"])
            loc_phrase = [prep, lpre, (text, loc_key), lpost]
    elif r.random() < 0.25:
        loc_phrase = [r.choice(["here", "in this folder", "in the current directory", "in this directory", "in here", "in the current folder"])]
        if r.random() < 0.12:
            flat = True
            loc_phrase = [r.choice(["in this folder only", "here but not in subfolders", "in this directory, not recursively",
                                    "just here, no subfolders", "only at the top level here"])]
    # noun
    if ext_noun:
        noun = [ext_noun]
    else:
        base = (["folders", "directories", "subfolders", "subdirectories", "dirs", "folders"] if dirs
                else ["files"] * 7 + ["items", "stuff", "things", "anything", "everything", "content", "data"])
        w = r.choice(base)
        if w in MASS and (sup_phrase or singular or pre):
            w = "files"
        if not dirs and sup_phrase and r.random() < 0.12:
            w = r.choice(["things", "ones", "items"])
        if singular:
            w = {"folders": "folder", "directories": "directory", "subfolders": "subfolder", "subdirectories": "subdirectory",
                 "dirs": "dir", "files": "file", "items": "item", "things": "thing", "ones": "one"}[w]
        noun = [w]
    if singular and ext_noun:
        singular = False; take = None if take == "none" else take
        if take is None:
            pass
    quant = "" if sup_phrase else r.choice(["all", "all the", "every", "any", "", "", "", "the", "all of the", "each", "all", "all my", "my", "those", "the", "any of the"])
    if noun and noun[0] in MASS:
        quant = r.choice(["", "", "all the", "the", "all that"]) if noun[0] in ("stuff", "things", "content", "data") else ""
    adjective = [r.choice(NEUTRAL)] if not sup_phrase and r.random() < 0.06 and not (noun and noun[0] in MASS) else []
    if quant in ("every", "each") and not ext_noun and not singular:
        noun = [{"files": "file", "folders": "folder", "directories": "directory", "subfolders": "subfolder",
                 "subdirectories": "subdirectory", "dirs": "dir", "items": "item"}.get(noun[0], noun[0])]
        if r.random() < 0.3:
            quant = r.choice(["every single", "each and every", "every last"])
    if quant in ("every", "each") and ext_noun:
        quant = "all"
    np = [quant] + (sup_phrase or []) + adjective
    for p in pre:
        np += p
    np += noun
    # sink-specific pieces
    dest_key, dest_phrase, fmt, arch_key = None, None, None, None
    if sink in ("move", "copy"):
        text, dpre, dpost = q.folder()
        dest_key = q.key()
        dest_phrase = [r.choice(["to", "into", "to", "over to", "in"] if sink == "move" else ["to", "into", "to", "in", "over to"]),
                       dpre, (text, dest_key), dpost]
    if sink == "compress":
        fmt = r.choice(["tgz", "tgz", "zip"])
        if r.random() < 0.6:
            base = q.word() + r.choice(["", "", "-backup", "_archive", "2024"])
            name = base + r.choice([".zip"] if fmt == "zip" else [".tar.gz", ".tgz", ".tar.gz", ""]) if r.random() < 0.75 else base
            if name.endswith(".zip"):
                fmt = "zip"
            arch_key = q.key()
            dest_phrase = [r.choice(["into", "as", "to", "called", "named", "into a file called", "and call it", "into an archive called"]),
                           (name, arch_key)]
    tailbits = list(post)
    if loc_phrase:
        pos = r.choices([len(tailbits), 0, r.randint(0, len(tailbits))], [6, 2, 2])[0]
        tailbits.insert(pos, loc_phrase)
    lead_loc = None
    if loc_phrase and r.random() < 0.06:
        tailbits.remove(loc_phrase)
        lead_loc = loc_phrase

    # ---- assemble English
    if r.random() < 0.28:
        q.lit(r.choice(POLITE))
    if lead_loc:
        emit(q, lead_loc); q.lit(",")
    style = r.random()
    if sink == "fcount":
        form = r.randrange(7)
        if form == 0:
            q.lit("how many"); emit(q, np[1:]); tail(q, tailbits); q.lit(r.choice(["are there", "", "exist", "do i have", "are there"]))
        elif form == 1:
            q.lit("how many"); emit(q, np[1:]); q.lit(r.choice(["are", "are there", "do i have", "exist", "live"])); tail(q, tailbits)
        elif form == 2:
            q.lit(r.choice(["count", "count up", "count all", "tally", "count how many"])); emit(q, np); tail(q, tailbits)
        elif form == 3:
            q.lit(r.choice(["tell me how many", "i want to know how many", "find out how many", "show how many"])); emit(q, np[1:]); tail(q, tailbits)
            q.lit(r.choice(["there are", "", "exist"]))
        elif form == 4:
            q.lit(r.choice(["number of", "the number of", "give me the number of", "what's the number of", "total number of", "count of"]))
            emit(q, np[1:]); tail(q, tailbits)
        elif form == 5:
            q.lit(r.choice(["what is the count of", "get the count of", "how many"])); emit(q, np[1:]); tail(q, tailbits)
        else:
            emit(q, np[1:]); tail(q, tailbits); q.lit(r.choice(["- how many", "how many are there", "count", "how many"]))
    elif sink == "total":
        form = r.randrange(6)
        if form == 0:
            q.lit(r.choice(["how much space do", "how much disk space do", "how much room do", "how much storage do"])); emit(q, np); tail(q, tailbits)
            q.lit(r.choice(["take", "take up", "use", "occupy", "use up", "eat"]))
        elif form == 1:
            q.lit(r.choice(["total size of", "what's the total size of", "the combined size of", "what is the combined size of",
                            "add up the size of", "sum the sizes of", "get the total size of", "overall size of"])); emit(q, np); tail(q, tailbits)
        elif form == 2:
            q.lit(r.choice(["how big are", "how large are", "how heavy are"])); emit(q, np); tail(q, tailbits)
            q.lit(r.choice(["in total", "altogether", "combined", "all together", "in total"]))
        elif form == 3:
            q.lit(r.choice(["how much disk space is used by", "how much space is taken by", "how much space is wasted on", "space used by"]))
            emit(q, np); tail(q, tailbits)
        elif form == 4:
            q.lit(r.choice(["what do", "how much do"])); emit(q, np); tail(q, tailbits); q.lit(r.choice(["add up to", "weigh in total", "total"]))
        else:
            q.lit(r.choice(["show", "tell me", "give me", "calculate", "work out"])); q.lit(r.choice(["the total size of", "how much space is used by", "the combined size of"]))
            emit(q, np); tail(q, tailbits)
    elif sink == "sizes":
        form = r.randrange(6)
        if dirs and form < 2 and not stages and not sup_phrase:
            q.lit(r.choice(["what's taking up space", "what is using the most space", "which folders are taking up space",
                            "what's eating my disk space", "where is my disk space going", "what is hogging disk space",
                            "which folders are the biggest", "show what's using space", "disk usage by folder"]))
            tail(q, tailbits)
        elif form < 4:
            q.lit(r.choice(["show the size of", "how big is", "what's the size of", "show sizes of", "list the sizes of",
                            "give me the size of", "how large is", "print the size of", "show how big", "sizes of"]))
            emit(q, np if np[0] else ["each"] + np[1:]); tail(q, tailbits)
        else:
            q.lit(r.choice(SHOW[:12])); emit(q, np); tail(q, tailbits)
            q.lit(r.choice(["with their sizes", "and their sizes", "along with how big they are", "with sizes", "and how much space each takes", "with size info"]))
    elif sink == "fshow":
        form = r.random()
        if form < 0.62:
            q.lit(r.choice(SHOW)); emit(q, np); tail(q, tailbits)
        elif form < 0.78:
            q.lit(r.choice(["what", "which"])); emit(q, np[1:]); q.lit(r.choice(["are", "are there", "do i have", "exist", "are sitting", ""])); tail(q, tailbits)
        elif form < 0.9:
            emit(q, np); tail(q, tailbits)                                   # bare noun phrase
        else:
            q.lit(r.choice(["are there any", "do i have any", "is there any", "any", "got any"])); emit(q, np[1:]); tail(q, tailbits)
    elif sink in FRAMES:
        frame_args = frame_emit(q, sink, np, tailbits)
    else:
        verb = r.choice({"delete": DELETE, "move": MOVE, "copy": COPY, "compress": ZIP if fmt == "zip" else TGZ}[sink])
        if fmt == "tgz" and arch_key is None and r.random() < 0.0:
            pass
        late = sink in ("move", "copy", "compress") and dest_phrase and r.random() < 0.12
        q.lit(verb); emit(q, np)
        if dest_phrase and not late and r.random() < 0.25:
            emit(q, dest_phrase); dest_phrase = None
        tail(q, tailbits)
        if dest_phrase:
            emit(q, dest_phrase)
    if sort_clause:
        q.lit(r.choice([",", "", "and show them", "-"]) if r.random() < 0.5 else ""); q.lit(sort_clause)
    if r.random() < 0.1:
        q.lit(r.choice(TAILS))

    # ---- program in canonical order
    pipe = [("files", {"what": "dirs" if dirs else "anyfile", "in": loc_key})]
    if flat:
        stages["flat"] = {}
    pipe += filter_stages(stages)
    if sort:
        pipe.append(("fsort", {"key": sort[0], "order": sort[1]}))
    if pick:
        pipe.append(("fpick", {"n": pick}))
    elif take:
        pipe.append(("ftake", {"n": None if take == "none" else take}))
    if sink in FRAMES:
        return pipe + frame_stages(sink, frame_args)
    args = {}
    if sink in ("move", "copy"):
        args = {"to": dest_key}
    if sink == "compress":
        args = {"fmt": fmt, "to": arch_key}
    pipe.append((sink, args))
    return pipe


# --------------------------------------------------------------------------- line requests
def lines_request(q):
    r = q.r
    f = q.filename(["log", "txt", "csv", "md", "json", "conf", "py", "sh", "out", "err", "tsv", "yaml"])
    fk = q.key()
    fa = (f, fk)
    ofs = ["in", "of", "from", "inside"]
    kind = r.choices(["grep", "head", "tail", "count", "sort", "uniq", "combo", "nomatch"], [30, 12, 14, 12, 8, 8, 10, 6])[0]
    pipe = [("lines", {"of": fk})]
    if r.random() < 0.25:
        q.lit(r.choice(POLITE))
    if kind in ("grep", "nomatch"):
        t = r.choice(PHRASES) if r.random() < 0.6 else q.word()
        k = q.key(); ta = (t, k, " " in t or r.random() < 0.2)
        if kind == "grep":
            form = r.randrange(9)
            emit(q, [[r.choice(["show", "find", "get", "print", "list", "show me", "display"]), r.choice(["lines", "all lines", "the lines", "every line"]),
                      r.choice(["containing", "with", "that contain", "that mention", "matching", "that include", "that have"]), ta, r.choice(ofs), fa],
                     ["grep", r.choice(["for", ""]), ta, "in", fa],
                     ["search", fa, "for", ta],
                     ["search for", ta, r.choice(["in", "inside", "within"]), fa],
                     ["find", ta, "in", fa],
                     [r.choice(["which lines", "what lines"]), r.choice(["in", "of"]), fa, r.choice(["contain", "mention", "have", "include", "match"]), ta],
                     ["look for", ta, "in", fa],
                     [r.choice(["where does", "where is"]), ta, r.choice(["appear in", "show up in", "occur in", "mentioned in"]), fa],
                     [r.choice(["lines", "all lines"]), r.choice(["in", "from"]), fa, r.choice(["with", "containing", "that have"]), ta]][form])
            pipe.append(("match", {"text": k}))
        else:
            form = r.randrange(5)
            emit(q, [[r.choice(["show", "print", "get", "list"]), r.choice(["lines", "the lines", "all lines"]), r.choice(ofs), fa,
                      r.choice(["that don't contain", "without", "that do not mention", "not containing", "that don't have", "excluding", "that lack"]), ta],
                     [r.choice(["show", "print"]), fa, r.choice(["without the lines containing", "minus lines with", "but skip lines with", "except lines containing", "hiding lines with"]), ta],
                     [r.choice(["filter out", "remove", "drop", "hide", "exclude", "strip"]), r.choice(["lines with", "lines containing", "lines that mention", "every line with"]), ta, "from", fa],
                     ["everything in", fa, r.choice(["except lines with", "that doesn't mention", "but not lines containing"]), ta],
                     ["grep -v", ta, fa]][form])
            pipe.append(("nomatch", {"text": k}))
        if r.random() < 0.12:
            q.lit(r.choice(["and count them", "and tell me how many", "- how many are there", "and count"]))
            pipe.append(("lcount", {}))
        elif r.random() < 0.1:
            out = q.filename(["txt", "log", "out"]); ok = q.key()
            emit(q, [r.choice(["and save them to", "and write them to", "into", "and put them in", "and save to", "> "]), (out, ok)])
            pipe.append(("save", {"to": ok}))
        else:
            pipe.append(("lshow", {}))
    elif kind in ("head", "tail"):
        n = q.count(); k = q.key()
        has_n = r.random() < 0.8
        na = (n, k)
        if kind == "head":
            forms = [[r.choice(["show", "print", "get", "display", "give me", "show me"]), "the first", na, "lines", r.choice(ofs), fa],
                     ["first", na, "lines", r.choice(ofs), fa], ["head", fa, na] if has_n else ["head", fa],
                     [r.choice(["top", "the top"]), na, "lines", r.choice(ofs), fa],
                     [r.choice(["the beginning of", "what's at the top of", "the top of", "the start of", "the opening lines of",
                                "peek at the top of", "what does the start of", "the first few lines of", "the first lines of", "how does"]), fa],
                     [r.choice(["show", "print"]), "the", r.choice(["start", "beginning", "top", "head"]), "of", fa],
                     ["what are the first", na, "lines", "of", fa]]
            form = r.randrange(len(forms))
            used_n = form in (0, 1, 3, 6) or (form == 2 and has_n)
            emit(q, forms[form])
            pipe.append(("ltake", {"n": k if used_n else None}))
        else:
            forms = [[r.choice(["show", "print", "get", "display", "give me", "show me"]), "the last", na, "lines", r.choice(ofs), fa],
                     ["last", na, "lines", r.choice(ofs), fa], ["tail", fa, na] if has_n else ["tail", fa],
                     [r.choice(["bottom", "the bottom", "the final", "final"]), na, "lines", r.choice(ofs), fa],
                     [r.choice(["the end of", "what's at the end of", "the bottom of", "the tail of", "tail of", "the latest lines in",
                                "what was logged last in", "the last few lines of", "the last lines of", "the most recent lines in"]), fa],
                     [r.choice(["show", "print"]), "the", r.choice(["end", "tail", "bottom", "last part"]), "of", fa],
                     ["what are the last", na, "lines", "of", fa], ["latest", na, "entries", r.choice(["in", "from"]), fa]]
            form = r.randrange(len(forms))
            used_n = form in (0, 1, 3, 6, 7) or (form == 2 and has_n)
            emit(q, forms[form])
            pipe.append(("llast", {"n": k if used_n else None}))
        pipe.append(("lshow", {}))
    elif kind == "count":
        form = r.randrange(7)
        emit(q, [["how many lines", r.choice(["are in", "in", "does", "are there in"]), fa, ],
                 ["count", r.choice(["the lines", "lines", "the number of lines"]), r.choice(ofs), fa],
                 ["how long is", fa, r.choice(["in lines", "", "(lines)"])], ["line count", r.choice(["of", "for"]), fa],
                 ["number of lines in", fa], ["wc -l", fa], [r.choice(["tell me", "show"]), "how many lines", fa, "has"]][form])
        pipe.append(("lcount", {}))
    elif kind == "sort":
        key = r.choice([("alpha", "asc"), ("alpha", "asc"), ("alpha", "desc"), ("num", "asc"), ("num", "desc")])
        how = {("alpha", "asc"): ["", "alphabetically", "in alphabetical order", "a to z", ""],
               ("alpha", "desc"): ["in reverse", "in reverse order", "backwards", "z to a", "in reverse alphabetical order", "descending"],
               ("num", "asc"): ["numerically", "by number", "as numbers", "in numeric order"],
               ("num", "desc"): ["numerically in reverse", "by number descending", "from highest to lowest number", "numerically, biggest first"]}[key]
        form = r.randrange(4)
        emit(q, [["sort", r.choice(["the lines of", "the lines in", "", "the contents of"]), fa, r.choice(how)],
                 ["sort", fa, r.choice(how)], [r.choice(["show", "print", "give me"]), fa, "sorted", r.choice(how)],
                 [r.choice(["order", "arrange"]), "the lines", r.choice(ofs), fa, r.choice(how)]][form])
        pipe.append(("lsort", {"key": key[0], "order": key[1]}))
        if r.random() < 0.15:
            out = q.filename(["txt", "csv", "out"]); ok = q.key()
            emit(q, [r.choice(["and save to", "and write the result to", "into", "and save it as"]), (out, ok)])
            pipe.append(("save", {"to": ok}))
        else:
            pipe.append(("lshow", {}))
    elif kind == "uniq":
        form = r.randrange(6)
        emit(q, [[r.choice(["remove", "drop", "delete", "strip", "get rid of"]), r.choice(["duplicate lines", "duplicates", "repeated lines", "dupes"]), r.choice(["from", "in"]), fa],
                 [r.choice(["show", "list", "print", "get"]), r.choice(["unique lines", "the unique lines", "distinct lines", "each line once"]), r.choice(ofs), fa],
                 ["dedupe", fa], ["deduplicate", r.choice(["", "the lines in", "lines of"]), fa],
                 [fa, r.choice(["without duplicates", "with duplicates removed", "but no repeated lines", "deduplicated"])],
                 ["unique", r.choice(["lines", "entries", "values"]), r.choice(["in", "from"]), fa]][form])
        pipe.append(("uniq", {}))
        if r.random() < 0.2:
            out = q.filename(["txt", "csv", "out"]); ok = q.key()
            emit(q, [r.choice(["and save to", "and write them to", "into", "and store in"]), (out, ok)])
            pipe.append(("save", {"to": ok}))
        elif r.random() < 0.1:
            q.lit(r.choice(["and count them", "- how many are there", "and tell me how many"]))
            pipe.append(("lcount", {}))
        else:
            pipe.append(("lshow", {}))
    else:  # combo: match -> sort -> uniq -> take -> save
        t = r.choice(PHRASES) if r.random() < 0.6 else q.word()
        k = q.key(); ta = (t, k, " " in t or r.random() < 0.2)
        emit(q, [r.choice(["get", "find", "take", "grab", "pull", "extract"]), r.choice(["lines", "the lines", "all lines", "every line"]),
                 r.choice(["with", "containing", "that mention", "matching"]), ta, r.choice(["from", "in"]), fa])
        pipe.append(("match", {"text": k}))
        steps = []
        if r.random() < 0.7:
            key = r.choice([("alpha", "asc"), ("alpha", "asc"), ("alpha", "desc"), ("num", "desc")])
            steps.append((r.choice({("alpha", "asc"): ["sort them", "sort", "put them in order", "sort alphabetically"],
                                    ("alpha", "desc"): ["sort them in reverse", "reverse sort them", "sort descending"],
                                    ("num", "desc"): ["sort numerically descending", "sort by number biggest first"]}[key]),
                          ("lsort", {"key": key[0], "order": key[1]})))
        if r.random() < 0.6:
            steps.append((r.choice(["remove duplicates", "drop duplicates", "dedupe", "keep unique ones", "remove repeats"]), ("uniq", {})))
        if r.random() < 0.35:
            n = q.count(); nk = q.key()
            steps.append(([r.choice(["keep the first", "take the first", "keep only the top", "show the first"]), (n, nk)], ("ltake", {"n": nk})))
        for i, (phrase, stage) in enumerate(steps):
            q.lit(r.choice([",", "and", "then", ", then", "and then"]))
            emit(q, phrase if isinstance(phrase, list) else [phrase])
            pipe.append(stage)
        if r.random() < 0.5:
            out = q.filename(["txt", "log", "out", "csv"]); ok = q.key()
            emit(q, [r.choice(["and save to", "and write to", "then save as", "and put the result in", "and save the result to"]), (out, ok)])
            pipe.append(("save", {"to": ok}))
        elif r.random() < 0.2:
            q.lit(r.choice(["and count them", "and count what's left", "then count"]))
            pipe.append(("lcount", {}))
        else:
            pipe.append(("lshow", {}))
    if r.random() < 0.08:
        q.lit(r.choice(TAILS))
    return pipe


# --------------------------------------------------------------------------- process requests
def procs_request(q):
    r = q.r
    pipe = [("procs", {})]
    kind = r.choices(["top", "kill_name", "kill_top", "show_name", "count_name", "list", "count"], [26, 24, 8, 16, 8, 12, 6])[0]
    if r.random() < 0.22:
        q.lit(r.choice(POLITE))
    procw = r.choice(["processes", "programs", "apps", "tasks", "processes", "things", "procs"])
    if kind == "top":
        key = r.choice(["mem", "mem", "cpu", "cpu"])
        res = {"mem": ["memory", "ram", "mem", "the most memory", "the most ram"], "cpu": ["cpu", "the cpu", "processor", "the most cpu", "cpu time"]}[key]
        pipe.append(("psort", {"key": key, "order": "desc"}))
        form = r.randrange(9)
        if form < 4:
            n = q.count(); k = q.key()
            emit(q, [[r.choice(["show", "list", "what are", "give me", "display"]), "the top", (n, k), procw, "by", r.choice(res[:3])],
                     ["top", (n, k), r.choice(res[:3]), r.choice(["hogs", "users", "consumers", "eaters"])],
                     [r.choice(["which", "what"]), (n, k), procw, r.choice(["use", "are using", "eat", "take"]), r.choice(["the most " + res[0], "the most " + res[1]] if key == "mem" else ["the most cpu", "the most processor"])],
                     [r.choice(["show", "list"]), "the", (n, k), procw, r.choice(["using the most", "with the highest", "eating the most"]), r.choice(res[:3])]][form])
            pipe.append(("ptake", {"n": k}))
        elif form < 7:
            emit(q, [[r.choice(["what's using", "what is using", "which process is using", "what's eating", "what is hogging", "who is using"]),
                      r.choice(["the most " + res[0], "all my " + res[0], "all the " + res[0], "so much " + res[0]])],
                     [r.choice(["which process is", "what's", "what is"]), r.choice(["hogging", "eating", "maxing out", "burning"]), r.choice(["the " + res[0], "my " + res[0]])],
                     [r.choice(["biggest", "top", "worst"]), res[0], r.choice(["hog", "user", "consumer"])]][form - 4])
            pipe.append(("ptake", {"n": None}))
        else:
            emit(q, [[r.choice(["show", "list"]), procw, r.choice(["sorted by", "ordered by", "by"]), r.choice(res[:3]), r.choice(["usage", "use", ""])],
                     [procw, r.choice(["by", "sorted by"]), r.choice(res[:3]), r.choice(["usage", ""])]][form - 7])
        pipe.append(("pshow", {}))
    elif kind in ("kill_name", "show_name", "count_name"):
        name = r.choice(PROCS) if r.random() < 0.7 else q.word()
        k = q.key(); na = (name, k)
        pipe.append(("pname", {"name": k}))
        if kind == "kill_name":
            force = r.random() < 0.3
            soft = ["kill", "stop", "terminate", "close", "end", "shut down", "quit", "kill all", "stop all", "exit", "pkill"]
            hard = ["force kill", "force quit", "kill -9", "forcibly kill", "hard kill", "force close", "nuke", "murder", "force stop"]
            verb = r.choice(hard if force else soft)
            form = r.randrange(5)
            emit(q, [[verb, na], [verb, na, r.choice(["processes", "process", "", "instances"])], [verb, "the", na, r.choice(["process", "app", "program"])],
                     [verb, r.choice(["all", "every", "any"]), na, r.choice(["processes", "process" if False else "processes", "instances"])],
                     [verb, r.choice(["anything named", "everything called", "processes named", "the process called"]), na]][form])
            pipe.append(("kill", {"force": "hard" if force else "soft"}))
        elif kind == "show_name":
            form = r.randrange(7)
            emit(q, [["is", na, r.choice(["running", "running?", "still running", "up", "alive", "active"])],
                     [r.choice(["show", "list", "find"]), na, r.choice(["processes", "process", "instances"])],
                     [r.choice(["show", "list", "find"]), r.choice(["processes named", "processes called", "all processes matching", "any process named"]), na],
                     ["check if", na, r.choice(["is running", "is up", "is alive", "is still going"])],
                     [r.choice(["what", "which"]), na, "processes are", r.choice(["running", "active", "alive"])],
                     ["pgrep", na], ["look for", r.choice(["a process called", "processes named", "the process"]), na]][form])
            pipe.append(("pshow", {}))
        else:
            form = r.randrange(4)
            emit(q, [["how many", na, r.choice(["processes", "instances"]), r.choice(["are running", "are there", "do i have", ""])],
                     ["count", r.choice(["the", ""]), na, r.choice(["processes", "instances"])],
                     ["number of", r.choice(["running", ""]), na, r.choice(["processes", "instances"])],
                     ["how many copies of", na, r.choice(["are running", "are open", "are there"])]][form])
            pipe.append(("pcount", {}))
    elif kind == "kill_top":
        key = r.choice(["mem", "cpu"])
        res = {"mem": ["memory", "ram"], "cpu": ["cpu", "processor"]}[key]
        pipe.append(("psort", {"key": key, "order": "desc"}))
        force = r.random() < 0.25
        verb = r.choice(["force kill", "kill -9", "forcibly kill"] if force else ["kill", "stop", "terminate", "end"])
        if r.random() < 0.5:
            n = q.count(); k = q.key()
            emit(q, [verb, "the", (n, k), procw, r.choice(["using the most", "eating the most", "with the highest", "hogging the most"]), r.choice(res)])
            pipe.append(("ptake", {"n": k}))
        else:
            emit(q, [[verb, r.choice(["the process", "whatever is", "the thing", "the app"]), r.choice(["using the most", "eating the most", "hogging the most", "that uses the most"]), r.choice(res)],
                     [verb, "the", r.choice(["biggest", "top", "worst"]), r.choice(res), r.choice(["hog", "user"])]][r.randrange(2)])
            pipe.append(("ptake", {"n": None}))
        pipe.append(("kill", {"force": "hard" if force else "soft"}))
    elif kind == "list":
        q.lit(r.choice(["show running processes", "list all processes", "what's running", "show me everything that's running",
                        "list processes", "what processes are running", "ps", "show all running programs", "which programs are running",
                        "running processes", "what is currently running", "show the process list", "display running tasks"]))
        pipe.append(("pshow", {}))
    else:
        q.lit(r.choice(["how many processes are running", "count running processes", "number of processes", "how many programs are running",
                        "how many things are running right now", "process count", "total number of running processes"]))
        pipe.append(("pcount", {}))
    if r.random() < 0.08:
        q.lit(r.choice(TAILS))
    return pipe


# --------------------------------------------------------------------------- stand-alone commands
def pkg_list(r, names):
    """'a b c', 'a and b', 'a, b and c' ... (the compiler drops the joining words)"""
    if len(names) == 1:
        return names[0]
    style = r.random()
    if style < 0.4:
        return " ".join(names)
    if style < 0.75:
        return " ".join(names[:-1]) + " and " + names[-1] if len(names) == 2 or r.random() < 0.5 else ", ".join(names[:-1]) + " and " + names[-1]
    return ", ".join(names)


def file_question(q):
    """Questions about files that do not start with a verb: 'does anything in src mention TODO', 'is there a file called x'."""
    r = q.r
    if r.random() < 0.2:
        q.lit(r.choice(POLITE))
    pipe = [("files", {"what": "anyfile", "in": None})]
    ext = None
    np = [r.choice(["anything", "any file", "any files", "any of the files", "something", "a file", "any document"])]
    if r.random() < 0.4:
        e = r.choice(EXT_NAMES) if r.random() < 0.5 else r.choice(EXTS)
        ext = q.key()
        np = [r.choice(["any", "any of the", "the", "any of my"]), (e, ext), "files"]
    loc = None
    locp = []
    if r.random() < 0.7:
        text, pre, post = q.folder()
        loc = q.key()
        locp = [r.choice(["in", "under", "inside", "in", "within"]), pre, (text, loc), post]
    pipe[0][1]["in"] = loc
    if ext:
        pipe.append(("ext", {"ext": ext}))
    kind = r.random()
    if kind < 0.6:
        t = r.choice(PHRASES) if r.random() < 0.5 else r.choice(WORDS + ["api_key", "TODO", "secret", "localhost", "username"])
        k = q.key(); ta = (t, k, " " in t)
        verb = r.choice(["mention", "contain", "include", "have", "reference", "use", "talk about", "refer to"])
        form = r.randrange(5)
        if form == 0:
            emit(q, [r.choice(["does", "do", "did"])] + np + locp + [verb, ta])
        elif form == 1:
            emit(q, [r.choice(["which files", "what files", "which of the files", "which file", "what file"])] + locp + [verb + r.choice(["", "s"]), ta])
            pipe = [pipe[0]] + [("contains", {"text": k})] if not ext else pipe
        elif form == 2:
            emit(q, ["is", ta, r.choice(["mentioned", "used", "referenced", "present", "anywhere"])] + (locp or [r.choice(["in any file", "anywhere here", "in any of the files"])]))
        elif form == 3:
            emit(q, [r.choice(["where is", "where do we use", "where does", "who uses", "where did i write"]), ta] + [r.choice(["used", "mentioned", "appear", "come up", "show up", ""])] + locp)
        else:
            emit(q, [r.choice(["search", "search all files", "look through", "scan", "grep through", "search everything"])] + locp[1:] + ["for", ta])
        if form == 1 and ext:
            pipe = [pipe[0]]
        if form in (3, 4):                               # these two phrasings never mention a file type
            return [pipe[0], ("fgrep", {"text": k})]
        if not any(st[0] == "contains" for st in pipe):
            if form in (1, 2) and ext:
                pipe = [pipe[0]]
            pipe.append(("contains", {"text": k}))
    else:
        name = q.filename()
        k = q.key()
        emit(q, [[r.choice(["is there", "do i have", "have i got", "is there still"]), r.choice(["a file", "a file", "any file", "something"]), r.choice(["called", "named"]), (name, k)] + locp,
                 [r.choice(["where is", "where's", "where did i put", "find", "locate", "where did"]), (name, k)] + locp,
                 [r.choice(["does", "is"]), (name, k), r.choice(["exist", "exist anywhere", "still exist", "around", "there"])] + locp][r.randrange(3)])
        pipe = [pipe[0], ("name", {"mode": "is", "text": k})]
    pipe.append(("fshow", {}))
    return pipe


JOIN = ["and then", ", then", "and", "then", "; then", ", and then", "and after that", ", after that", "and afterwards", "; ", "and next", ", next"]


def mkdir_then(q):
    """'make a folder called X and move all pdfs into it': the second step points back at X."""
    r = q.r
    name = q.word() + r.choice(["", "", "-backup", "_old", "2024"])
    if r.random() < 0.3:
        name = f"{r.choice(REAL_DIRS)}/{name}"
    k = q.key()
    if r.random() < 0.2:
        q.lit(r.choice(POLITE))
    emit(q, [r.choice(["make a folder called", "create a directory called", "create a folder named", "mkdir", "make a new folder", "create folder",
                       "make a directory named", "create a new directory called"]), (name, k)])
    q.lit(r.choice(JOIN[:8]))
    stages, pre, post, ext_noun = file_filters(q, False)
    sink = r.choice(["move", "move", "copy"])
    verb = r.choice(MOVE if sink == "move" else COPY)
    np = [r.choice(["all", "all the", "every", "the", "", "all my"])]
    for p in pre:
        np += p
    np += [ext_noun] if ext_noun else ["files"]
    q.lit(verb); emit(q, np)
    for t in post:
        emit(q, t)
    loc = None
    if r.random() < 0.5:
        loc = q.folder_arg(["from", "in", "out of", "under"])
    back = r.random()
    if back < 0.7:
        q.lit(r.choice(["into it", "there", "to it", "into that folder", "in there", "into the new folder", "to that folder", "in it", "to the new folder"]))
        dest = k
    else:
        dest = q.key()
        emit(q, [r.choice(["to", "into"]), (name, dest)])
    pipe = [("files", {"what": "anyfile", "in": loc})] + filter_stages(stages)
    pipe.append((sink, {"to": dest}))
    return [[("mkdir", {"path": k})], pipe]


def typo(r, word):
    if len(word) < 4 or not word.isalpha():
        return word
    i = r.randrange(1, len(word) - 1)
    kind = r.randrange(3)
    if kind == 0:
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if kind == 1:
        return word[:i] + word[i + 1:]
    return word[:i] + word[i] + word[i:]
