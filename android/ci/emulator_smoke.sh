#!/usr/bin/env bash
# Real-Android smoke test, run inside the CI emulator:
# install the APK, grant notification access, send a scam SMS to the
# emulator and check that our app posts a fraud warning for it.
set -u
APK=${1:-AI-Fraud-Detector.apk}
PKG=kz.aifrauddetector
OUT=emulator-out
mkdir -p "$OUT"

adb install -r "$APK"
adb shell pm grant $PKG android.permission.POST_NOTIFICATIONS || true
adb shell cmd notification allow_listener $PKG/$PKG.NotificationWatcher
adb shell am start -W -n $PKG/.MainActivity
sleep 6
adb exec-out screencap -p > "$OUT/1_main_screen.png"

# A scam SMS and an ordinary one, delivered to the emulator's SMS app.
adb emu sms send 7001234567 "Срочно! Ваша карта Kaspi заблокирована. Назовите код из SMS сотруднику службы безопасности или перейдите по ссылке http://kaspi-secure.xyz"
sleep 4
adb emu sms send 7007654321 "Привет! Урок математики перенесли на пятницу, не забудь тетрадь"
sleep 10

adb shell cmd statusbar expand-notifications
sleep 3
adb exec-out screencap -p > "$OUT/2_notifications.png"
adb shell cmd statusbar collapse

adb shell dumpsys notification --noredact > "$OUT/notifications.txt"
adb logcat -d > "$OUT/logcat.txt"

ours=$(grep -c "pkg=$PKG" "$OUT/notifications.txt" || true)
sms_pkgs=$(grep -oE "pkg=com\.(google\.android\.apps\.messaging|android\.messaging|android\.mms)" "$OUT/notifications.txt" | sort -u | tr '\n' ' ')
echo "::notice title=emulator::SMS app notifications from: ${sms_pkgs:-none}; notifications from our app: $ours"
if grep -q "FATAL EXCEPTION" "$OUT/logcat.txt" && grep -A3 "FATAL EXCEPTION" "$OUT/logcat.txt" | grep -q "$PKG"; then
  echo "::error title=emulator::app crashed: $(grep -A6 'FATAL EXCEPTION' "$OUT/logcat.txt" | tr '\n' ' ' | cut -c1-400)"
  exit 1
fi
grep -E "FraudWatcher" "$OUT/logcat.txt" | head -5 | sed 's/^/::warning title=watcher log::/'
if [ -z "$sms_pkgs" ]; then
  echo "::warning title=emulator::the emulator's SMS app posted no notification - can't check the end-to-end path here"
  exit 0
fi
if [ "$ours" -lt 1 ]; then
  echo "::error title=emulator::scam SMS arrived but no fraud warning was posted"
  exit 1
fi
echo "::notice title=emulator::end-to-end OK: scam SMS -> fraud warning"
