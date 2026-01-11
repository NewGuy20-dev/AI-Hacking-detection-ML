#!/bin/bash
# Setup cron jobs for automated training and stress testing

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PYTHON_PATH="${PYTHON_PATH:-python3}"
LOG_DIR="$PROJECT_DIR/logs"

# Create log directory
mkdir -p "$LOG_DIR"

echo "Setting up cron jobs for ML pipeline..."
echo "Project directory: $PROJECT_DIR"
echo "Python path: $PYTHON_PATH"

# Backup existing crontab
crontab -l > /tmp/crontab_backup 2>/dev/null || true

# Create new cron entries
cat << EOF > /tmp/ml_cron_jobs
# ML Pipeline Cron Jobs
# Generated on $(date)

# Daily stress test at 3:00 AM
0 3 * * * cd $PROJECT_DIR && $PYTHON_PATH src/stress_test/runner.py >> $LOG_DIR/stress_test.log 2>&1

# Weekly full training on Sunday at 1:00 AM
0 1 * * 0 cd $PROJECT_DIR && $PYTHON_PATH src/stress_test/train_pipeline.py --full >> $LOG_DIR/training.log 2>&1

# Clean old logs (keep 30 days) - runs daily at 4:00 AM
0 4 * * * find $LOG_DIR -name "*.log" -mtime +30 -delete 2>/dev/null

# Clean old reports (keep 30 days) - runs daily at 4:00 AM
0 4 * * * find $PROJECT_DIR/reports -name "*.json" -mtime +30 -delete 2>/dev/null

EOF

echo ""
echo "The following cron jobs will be added:"
echo "----------------------------------------"
cat /tmp/ml_cron_jobs
echo "----------------------------------------"
echo ""

read -p "Do you want to install these cron jobs? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Merge with existing crontab
    (crontab -l 2>/dev/null | grep -v "ML Pipeline" | grep -v "stress_test" | grep -v "train_pipeline"; cat /tmp/ml_cron_jobs) | crontab -
    echo "✅ Cron jobs installed successfully!"
    echo ""
    echo "Current crontab:"
    crontab -l
else
    echo "Cron jobs not installed."
    echo "You can manually add them by running: crontab -e"
fi

# Create systemd timer alternative (for systems that prefer systemd)
cat << 'EOF' > /tmp/ml-stress-test.service
[Unit]
Description=ML Model Daily Stress Test
After=network.target

[Service]
Type=oneshot
WorkingDirectory=${PROJECT_DIR}
ExecStart=${PYTHON_PATH} src/stress_test/runner.py
StandardOutput=append:${LOG_DIR}/stress_test.log
StandardError=append:${LOG_DIR}/stress_test.log

[Install]
WantedBy=multi-user.target
EOF

cat << 'EOF' > /tmp/ml-stress-test.timer
[Unit]
Description=Run ML Stress Test Daily at 3 AM

[Timer]
OnCalendar=*-*-* 03:00:00
Persistent=true

[Install]
WantedBy=timers.target
EOF

echo ""
echo "Systemd timer files created in /tmp/"
echo "To use systemd instead of cron:"
echo "  sudo cp /tmp/ml-stress-test.service /etc/systemd/system/"
echo "  sudo cp /tmp/ml-stress-test.timer /etc/systemd/system/"
echo "  sudo systemctl daemon-reload"
echo "  sudo systemctl enable --now ml-stress-test.timer"

# Cleanup
rm -f /tmp/ml_cron_jobs

echo ""
echo "Setup complete!"
