//! Value pools, copied from kural/gen.py. Arguments are drawn from these at random, so the model learns to point
//! at words rather than memorise them.

use std::sync::LazyLock;

fn split(s: &'static str) -> Vec<&'static str> {
    s.split_whitespace().collect()
}

pub static WORDS: LazyLock<Vec<&'static str>> = LazyLock::new(|| {
    split(
        "alpha beta gamma delta report draft notes invoice budget summary backup archive photo holiday
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
falcon eagle sparrow otter badger tiger panda koala lynx heron comet nova pulsar quasar nebula orbit",
    )
});
pub static EXTS: LazyLock<Vec<&'static str>> = LazyLock::new(|| {
    split("py js ts rs go c h cpp java rb sh md txt log csv json yaml yml xml html css pdf jpg jpeg png gif svg mp3 mp4 mkv mov wav flac zip tar gz bak tmp conf ini toml lock sql db iso deb exe docx xlsx")
});
pub static EXT_NAMES: &[&str] = &["python", "javascript", "typescript", "rust", "markdown", "text", "shell", "ruby", "golang", "java",
    "json", "yaml", "csv", "html", "css", "log", "pdf", "backup", "temp", "config", "c++", "kotlin"];
pub static EXT_PLURALS: &[&str] = &["pdfs", "logs", "images", "pictures", "photos", "videos", "movies", "songs", "archives", "zips", "jpgs",
    "pngs", "mp4s", "mp3s", "csvs", "isos", "gifs", "txts", "jsons", "backups"];
pub static FOLDER_NAMES: &[&str] = &["downloads", "documents", "desktop", "pictures", "music", "videos", "home", "tmp", "projects", "trash",
    "photos", "movies", "downloads", "documents", "desktop"];
pub static REAL_DIRS: &[&str] = &["~/Downloads", "~/Documents", "/var/log", "/etc", "/tmp", "~", "/usr/share", "~/.config", "/opt",
    "/home", "/var", "~/Desktop", "/usr/local/bin", "node_modules", "src", "tests", "build", "dist",
    "docs", "./build", "../data", "~/code", "/mnt/usb", "/srv/www", "~/.cache", "lib", "bin", "target"];
pub static DIR_WORDS: LazyLock<Vec<&'static str>> = LazyLock::new(|| {
    split(
        "assets packages www public static vendor scripts config data images media uploads logs cache backups
notes work dev code repos include templates migrations fixtures resources themes plugins modules components pages
styles fonts icons locales samples examples benchmarks tools utils archive old tmp temp output results reports
exports imports shared common core api server client frontend backend mobile desktop web app apps services",
    )
});
pub static SPECIAL_FILES: &[&str] = &["/etc/hosts", "/etc/fstab", "/etc/passwd", "Makefile", "Dockerfile", "README", "LICENSE", ".bashrc", "~/.bashrc",
    "~/.zshrc", ".gitignore", "/proc/cpuinfo", "/etc/os-release", ".env", "~/.ssh/config", "/var/log/syslog",
    "/var/log/messages", "Cargo.toml", "package.json", "requirements.txt", "CHANGELOG", "TODO", "notes",
    "/etc/resolv.conf", "~/.profile", "/var/log/auth.log", "docker-compose.yml", "/etc/ssh/sshd_config",
    "Gemfile", "go.mod", "pyproject.toml", "/etc/hostname", "~/.gitconfig", "/proc/meminfo", "Vagrantfile"];
pub static PKGS: LazyLock<Vec<&'static str>> = LazyLock::new(|| {
    split(
        "htop vim neovim git curl wget tmux fzf ripgrep bat fd jq tree rsync docker nodejs npm python-pip gcc make
cmake clang rustup go ffmpeg imagemagick gimp vlc mpv firefox chromium thunderbird libreoffice inkscape blender
nginx postgresql redis sqlite zsh fish neofetch btop ncdu unzip p7zip zip openssh nmap wireshark strace gdb
valgrind kdenlive obs-studio steam discord telegram-desktop qbittorrent keepassxc syncthing restic borg",
    )
});
pub static PROCS: LazyLock<Vec<&'static str>> = LazyLock::new(|| {
    split(
        "firefox chrome chromium node python python3 java nginx postgres redis docker code slack discord steam
spotify zoom vlc mpv bash zsh ssh sshd cron systemd plasmashell kwin electron thunderbird obs blender gimp
webpack cargo rustc gcc make npm yarn ruby php mysqld mongod telegram signal",
    )
});
pub static BRANCHES: &[&str] = &["main", "master", "dev", "develop", "staging", "feature/login", "feature/search", "fix-bug-12", "hotfix",
    "release-1.2", "experiment", "refactor", "wip", "testing", "docs-update", "v2", "feat/api", "bugfix/crash"];
pub static PHRASES: &[&str] = &["import numpy", "TODO", "FIXME", "error", "timeout", "password", "api key", "connection refused",
    "segfault", "deprecated", "hello world", "not found", "warning", "fatal", "http://", "localhost",
    "null pointer", "out of memory", "copyright", "license", "main()", "def test", "console.log", "panic",
    "failed", "denied", "exception", "traceback", "debug", "user id", "access token", "retry", "404"];
pub static MSGS: &[&str] = &["fix login bug", "initial commit", "update readme", "add tests", "refactor parser", "wip", "bump version",
    "fix typo", "add search feature", "remove dead code", "improve error handling", "update dependencies",
    "first draft", "cleanup", "handle empty input", "speed up startup", "add dark mode", "fix crash on exit"];
pub static NUM_WORDS: &[&str] = &["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "twelve", "fifteen", "twenty"];
pub static POLITE: &[&str] = &["please", "can you", "could you", "i want to", "i need to", "i'd like to", "help me", "let's", "go ahead and",
    "quickly", "now", "ok", "hey", "pls", "can u", "would you", "just", "i wanna", "time to", "kindly",
    "yo", "ok so", "alright", "need to", "want to", "gotta", "let me", "so"];
pub static TAILS: &[&str] = &["please", "for me", "thanks", "pls", "now", "if you can", "right now", "thank you", "asap", "ok", "quickly"];

pub static UNITS: LazyLock<Vec<&'static str>> = LazyLock::new(|| {
    split(
        "nginx sshd docker bluetooth NetworkManager cups cronie postgresql redis mysql apache2 httpd ufw firewalld
tailscaled libvirtd sddm gdm pipewire systemd-resolved fail2ban plex jellyfin mosquitto grafana prometheus ollama
syncthing tor mariadb mongodb elasticsearch rabbitmq caddy haproxy samba nfs-server chronyd avahi-daemon
wpa_supplicant ssh cron apache postgres containerd k3s vsftpd dnsmasq squid zerotier-one transmission",
    )
});
pub static HOSTS: &[&str] = &["10.0.0.5", "192.168.1.20", "example.com", "myserver", "prod-01", "github.com", "raspberrypi.local", "google.com",
    "8.8.8.8", "pi@192.168.0.12", "root@10.1.2.3", "deploy@staging.example.org", "nas", "db1.internal", "localhost",
    "archlinux.org", "user@host", "srv.example.net", "172.16.0.9", "homelab", "vps", "alice@build-box", "1.1.1.1",
    "wikipedia.org", "my-router", "192.168.0.1", "git@gitlab.com", "backup.lan", "node3", "admin@10.10.10.10"];
pub static URLS: &[&str] = &["https://example.com/file.zip", "http://releases.org/app.tar.gz", "https://github.com/user/repo", "example.org/data.csv",
    "https://github.com/torvalds/linux.git", "git@github.com:me/dotfiles.git", "https://api.example.com/v1/status",
    "http://localhost:8080/health", "https://cdn.site.net/img/logo.png", "https://get.tool.sh/install.sh",
    "https://files.example.io/dataset-2024.tar.xz", "localhost:3000", "https://gitlab.com/group/project.git",
    "http://192.168.1.5:9000/metrics", "https://example.org", "https://dl.example.net/iso/distro-24.04.iso"];
pub static ENVNAMES: LazyLock<Vec<&'static str>> = LazyLock::new(|| {
    split(
        "PATH HOME EDITOR JAVA_HOME API_KEY DEBUG NODE_ENV DATABASE_URL LANG TERM SHELL USER PYTHONPATH GOPATH
AWS_PROFILE PORT HOSTNAME XDG_CONFIG_HOME CC LD_LIBRARY_PATH TZ DISPLAY VISUAL OPENAI_API_KEY MY_VAR FOO BAR
SECRET_TOKEN LOG_LEVEL HTTP_PROXY CUDA_VISIBLE_DEVICES RUST_BACKTRACE PAGER BROWSER APP_ENV",
    )
});
pub static CMDNAMES: LazyLock<Vec<&'static str>> = LazyLock::new(|| {
    split(
        "tar ls grep rsync ffmpeg awk sed git docker curl ssh find chmod systemctl python node gcc make cargo npm
pip vim nvim tmux htop rg fd jq go java ruby php kubectl terraform ansible cmake clang zsh bash fish less cat cp mv dd
df du ps top xargs sort uniq cut tr wc python3 rustc deno bun yarn pnpm emacs nano code firefox mpv",
    )
});
pub static CONTAINERS: &[&str] = &["web", "db", "redis", "nginx", "app", "api", "worker", "my-app", "postgres", "backend", "frontend", "proxy",
    "cache", "3f2a9c1b", "a1b2c3d4e5f6", "mongo", "grafana", "traefik", "test-runner", "dev_db_1", "project-web-1"];
pub static IMAGES: &[&str] = &["nginx", "ubuntu:22.04", "python:3.12", "alpine", "redis:7", "postgres:16", "node:20", "user/app:latest",
    "debian:stable-slim", "golang:1.22", "mysql", "busybox", "ghcr.io/org/tool:v2", "archlinux", "myapp:dev"];
pub static SCRIPTS: &[&str] = &["build", "test", "start", "dev", "lint", "deploy", "watch", "format", "serve", "clean", "typecheck", "preview", "e2e", "docs", "migrate"];
pub static SSIDS: &[&str] = &["HomeWifi", "Office 5G", "eduroam", "NETGEAR-42", "Starbucks", "my hotspot", "TP-Link_5F3A", "guest", "Pixel 8",
    "CoffeeShop Free", "FRITZ!Box 7590", "lab-net", "AndroidAP", "Linksys01234"];
pub static OWNERS: &[&str] = &["root", "www-data", "alice", "bob", "nobody", "postgres", "me", "deploy", "http", "git", "myself", "carol",
    "admin", "ubuntu", "dave", "nginx", "mysql", "jenkins"];
pub static MODE_WORDS: &[&str] = &["executable", "read only", "read-only", "writable", "private", "public", "644", "755", "600", "700", "777",
    "400", "u+x", "go-w", "a+r", "+x", "o-rwx", "g+w", "0644", "readonly", "world readable", "664", "750"];
pub static DATES: &[&str] = &["2024-01-15", "jan 5", "last monday", "march 3rd", "last friday", "2023-12-31", "december 1", "june 2024",
    "oct 12 2023", "tuesday", "2025-06-30", "new year", "the 1st of may", "feb 14", "last sunday", "2024-11-01",
    "september 9th", "01/02/2024", "april", "last christmas", "wednesday", "aug 20 2022"];
pub static EXPRS: &[&str] = &["12*8", "1920 / 1080", "2^10", "(4+5)*3", "15 percent of 240", "7 times 6", "100 divided by 7", "3.14 * 2",
    "144/12", "2 plus 2", "1024*1024", "99 minus 45", "5 squared", "17 % 5", "60*60*24", "2 + 2", "365 * 24",
    "1000 / 3", "18 percent of 55", "(12 + 30) / 2", "8 times 125", "45 plus 78 plus 9"];
pub static REMOTES: &[&str] = &["pi@host:/backup", "user@server:~/uploads", "nas:/volume1/backup", "root@10.0.0.5:/srv", "backup-box:",
    "deploy@prod:/var/www", "me@vps:files/", "alice@192.168.1.9:~/incoming", "storage.lan:/data", "myserver:~",
    "ubuntu@ec2-host:/home/ubuntu", "nas:photos/", "web01:/var/www/html", "git@build:/artifacts"];
pub static PCTS: &[&str] = &["50", "50%", "30 percent", "80%", "100", "half", "max", "full", "25%", "10", "75 percent", "0", "60%", "40",
    "90%", "15", "100%", "20 percent", "5"];
pub static PORTS: &[&str] = &["80", "443", "8080", "3000", "5000", "8000", "22", "5432", "3306", "6379", "9090", "4200", "8888", "27017",
    "5173", "9000", "25565", "1234", "8443", "53"];
pub static AFFIXES: &[&str] = &["old_", "new-", "backup_", "2024-", "_final", "-v2", "_old", ".bak", "draft-", "IMG_", "copy_", "_edited",
    "holiday-", "x_", "-small", "_backup", "tmp-", "archive_"];
pub static ALIASES: &[&str] = &["ll", "gs", "k", "la", "gp", "dc", "py", "..", "cls", "gco", "update", "serve", "weather", "please", "tf", "v"];
pub static ALIAS_CMDS: &[&str] = &["ls -la", "git status", "kubectl", "git push", "docker compose", "python3", "cd ..", "clear",
    "git checkout", "sudo pacman -Syu", "python3 -m http.server", "sudo", "terraform", "nvim", "ls -lah --color"];
pub static VAR_VALUES: &[&str] = &["vim", "production", "1", "/usr/lib/jvm/java-17", "true", "development", "8080", "en_US.UTF-8", "nano",
    "debug", "/opt/tools/bin", "abc123", "0", "postgres://localhost/db", "false", "firefox", "42", "eu-west-1"];
pub static SENTENCES: &[&str] = &["hello world", "export EDITOR=vim", "TODO: fix this", "127.0.0.1 myhost", "done", "node_modules/", "*.log",
    "alias ll='ls -la'", "# end of file", "debug=true", "first line", "see you later", ".env", "foo", "bar baz",
    "set -e", "PATH=$PATH:~/bin", "this is a test", "remember the milk", "build/", "v1.0.0 released"];
pub static TAGS: &[&str] = &["v1.0", "v1.2.0", "v2.0.0-rc1", "release-2024", "stable", "v0.3", "beta", "1.0.0", "prod-2024-05", "nightly"];
pub static ORDINALS: &[&str] = &["second", "third", "2nd", "3rd", "fourth", "4th", "fifth", "5th", "first", "1st"];
pub static RAW_COMMANDS: &[&str] = &["ls -la", "git status", "docker ps", "cat /etc/hosts", "make", "npm test", "df -h", "python script.py",
    "./build.sh", "grep -r foo .", "ps aux", "pacman -Syu", "systemctl restart nginx", "ls", "cargo build",
    "kubectl get pods", "tail -n 50 /var/log/syslog", "ip a", "du -sh *", "find . -name '*.py'", "mount /dev/sdb1 /mnt",
    "dmesg", "journalctl -xe", "apt update", "nmap -sn 192.168.1.0/24", "rm -rf build", "cat notes.txt",
    "python3 manage.py migrate", "git log --oneline", "free -h", "lsblk", "curl -s https://example.com", "top -bn1",
    "cp a.txt b.txt", "mkdir -p out/logs", "ss -tulpn", "env", "history", "uname -a", "wc -l *.txt", "pytest -q",
    "docker compose up -d", "chmod 600 key.pem", "vim config.yaml", "tar -xf data.tar.gz", "echo $PATH"];

// ---- file requests
pub static MASS: &[&str] = &["stuff", "things", "anything", "everything", "content", "data"];
pub static NEUTRAL: &[&str] = &["random", "stray", "leftover", "unused", "junk", "useless", "extra", "various", "misc", "loose", "spare", "assorted"];
pub fn vague(n: &str) -> &'static [&'static str] {
    match n {
        "larger" => &["large", "big", "huge", "giant", "massive", "heavy", "bulky", "fat", "enormous", "really big", "very large"],
        "smaller" => &["small", "tiny", "little", "really small", "very small", "teeny"],
        "older" => &["old", "ancient", "stale", "outdated", "really old", "very old", "aging", "dusty"],
        _ => &["new", "recent", "fresh", "recently changed", "recently modified", "brand new"],
    }
}
pub fn superlative(key: (&str, &str)) -> &'static [&'static str] {
    match key {
        ("size", "desc") => &["largest", "biggest", "heaviest", "largest", "biggest", "bulkiest", "fattest"],
        ("size", "asc") => &["smallest", "tiniest", "lightest", "smallest"],
        ("time", "desc") => &["newest", "most recent", "latest", "most recently modified", "most recently changed",
            "most recently edited", "freshest", "most recently updated"],
        _ => &["oldest", "least recently modified", "least recently used", "earliest", "oldest"],
    }
}
pub static SORT_KEYS: &[(&str, &str)] = &[("size", "desc"), ("size", "asc"), ("time", "desc"), ("time", "asc"), ("name", "asc"), ("name", "desc")];
pub fn sort_clause(key: (&str, &str)) -> &'static [&'static str] {
    match key {
        ("size", "desc") => &["sorted by size", "biggest first", "largest first", "ordered by size", "from largest to smallest",
            "by size", "in order of size", "sorted by size descending", "with the biggest at the top"],
        ("size", "asc") => &["smallest first", "sorted by size ascending", "from smallest to largest", "ordered by size smallest first"],
        ("time", "desc") => &["newest first", "sorted by date", "most recent first", "ordered by modification time",
            "by date", "sorted by time", "latest first", "sorted by when they were changed"],
        ("time", "asc") => &["oldest first", "sorted by date oldest first", "from oldest to newest", "earliest first"],
        ("name", "asc") => &["sorted by name", "alphabetically", "in alphabetical order", "ordered by name", "sorted a to z"],
        _ => &["in reverse alphabetical order", "sorted by name descending", "sorted z to a"],
    }
}
pub static SHOW: &[&str] = &["show", "show me", "list", "find", "display", "print", "give me", "get", "look for", "search for", "locate",
    "find me", "list out", "i want to see", "let me see", "pull up", "find all", "list all", "show all",
    "what are", "which are", "where are", "get me", "dig up", "track down", "ls", "tell me", "look up", "see",
    "show me all", "can i see", "fetch", "get a list of", "give me a list of", "find any", "i need", "i need to see",
    "point me to", "bring up", "check for", "scan for", "hunt for"];
pub static DELETE: &[&str] = &["delete", "remove", "get rid of", "erase", "wipe", "clean up", "purge", "rm", "nuke", "clear out",
    "throw away", "delete all", "remove all", "wipe out", "destroy", "discard", "drop", "clean out", "kill off"];
pub static MOVE: &[&str] = &["move", "transfer", "relocate", "mv", "shift", "put", "move all", "send", "dump"];
pub static COPY: &[&str] = &["copy", "duplicate", "back up", "cp", "copy all", "make a copy of", "clone", "save a copy of", "backup"];
pub static TGZ: &[&str] = &["compress", "archive", "tar up", "make a tarball of", "pack", "bundle", "tar", "make an archive of", "gzip", "pack up"];
pub static ZIP: &[&str] = &["zip", "zip up", "make a zip of", "create a zip of", "zip together", "put in a zip"];
pub static CONNECT: &[&str] = &["and", ",", "but", "and also", "and", "that are also", "plus", "and are", ", and"];
pub static JOIN: &[&str] = &["and then", ", then", "and", "then", "; then", ", and then", "and after that", ", after that", "and afterwards", "; ", "and next", ", next"];
