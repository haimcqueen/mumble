#!/bin/zsh
# Build dist/Mumble.dmg — a drag-to-Applications disk image containing
# Mumble.app, a self-installing wrapper around the Python source.
#
# On first launch the app copies its bundled source to
# ~/Library/Application Support/Mumble, opens Terminal, runs install.sh
# (venv + deps + model download), then starts dictation. Later launches
# skip straight to dictation.
#
# NOTE: without an Apple Developer ID the app is unsigned — downloaders
# must right-click → Open the first time (see README).
set -e
cd "$(dirname "$0")"

VERSION=${1:-0.1.0}
APP=dist/Mumble.app
STAGING=dist/dmg-staging

rm -rf dist
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources/src"

# ---- bundle the source ------------------------------------------------
cp dictate.py dictate.sh install.sh requirements.txt README.md LICENSE \
   "$APP/Contents/Resources/src/"

# ---- Info.plist --------------------------------------------------------
cat > "$APP/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key>              <string>Mumble</string>
  <key>CFBundleDisplayName</key>       <string>Mumble</string>
  <key>CFBundleIdentifier</key>        <string>com.haimcqueen.mumble</string>
  <key>CFBundleVersion</key>           <string>$VERSION</string>
  <key>CFBundleShortVersionString</key><string>$VERSION</string>
  <key>CFBundlePackageType</key>       <string>APPL</string>
  <key>CFBundleExecutable</key>        <string>mumble</string>
  <key>LSMinimumSystemVersion</key>    <string>14.0</string>
  <key>NSMicrophoneUsageDescription</key>
  <string>Mumble listens while you dictate. Audio never leaves your Mac.</string>
</dict>
</plist>
EOF

# ---- launcher ----------------------------------------------------------
cat > "$APP/Contents/MacOS/mumble" <<'EOF'
#!/bin/zsh
# Mumble.app launcher: sync bundled source to Application Support, then
# run the installer/dictation inside a Terminal window (dictation is a
# terminal app; Terminal also owns the macOS permission grants).
set -e
HERE="$(cd "$(dirname "$0")/../Resources/src" && pwd)"
DEST="$HOME/Library/Application Support/Mumble"

mkdir -p "$DEST"
for f in dictate.py dictate.sh install.sh requirements.txt README.md LICENSE; do
  cp "$HERE/$f" "$DEST/"
done
chmod +x "$DEST/dictate.sh" "$DEST/install.sh"

BOOT="$DEST/run.command"
cat > "$BOOT" <<'BOOTEOF'
#!/bin/zsh
cd "$HOME/Library/Application Support/Mumble"
if [[ ! -d .venv ]]; then
  echo "First run — installing Mumble (a few minutes) ..."
  ./install.sh
fi
exec ./dictate.sh
BOOTEOF
chmod +x "$BOOT"
exec open -a Terminal "$BOOT"
EOF
chmod +x "$APP/Contents/MacOS/mumble"

# ---- disk image --------------------------------------------------------
mkdir -p "$STAGING"
cp -R "$APP" "$STAGING/"
ln -s /Applications "$STAGING/Applications"
cat > "$STAGING/READ ME FIRST.txt" <<'EOF'
MUMBLE — local, private dictation for macOS (Apple Silicon)

1. Drag Mumble.app into Applications.
2. IMPORTANT — the app is unsigned, so the FIRST time:
   right-click Mumble.app → Open → Open.
   (Double-clicking will show a warning instead.)
3. A Terminal window opens, installs everything on first run
   (needs internet once, ~600 MB model), then starts dictation.
4. Grant the three permissions it asks for, then:
   hold Fn → speak → release. Text appears where your cursor is.

Requires: Apple Silicon (M1+), macOS 14+, Python 3.10+
Optional cleanup: install Ollama (ollama.com) and run:
   ollama pull qwen2.5:7b

Everything runs on your Mac. No cloud. No account. Free forever.
Source: https://github.com/haimcqueen/mumble
EOF

hdiutil create -volname "Mumble" -srcfolder "$STAGING" -ov -format UDZO \
  "dist/Mumble-$VERSION.dmg" >/dev/null
rm -rf "$STAGING"

echo "built dist/Mumble-$VERSION.dmg"
du -h "dist/Mumble-$VERSION.dmg" | cut -f1
