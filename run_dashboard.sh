#!/bin/bash
# ============================================
# 🚀 One-Tap Trading Dashboard Launcher
# ============================================
# Usage: ./run_dashboard.sh
# ============================================

cd "$(dirname "$0")"

echo "============================================"
echo "🚀 Algorithmic Trading Dashboard"
echo "============================================"

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "❌ No venv found. Run: python -m venv venv && pip install -r requirements.txt"
    exit 1
fi

# Check if dashboard data exists, generate if not
DATA_FILE="data/dashboard/dash_data.pkl"
if [ ! -f "$DATA_FILE" ]; then
    echo ""
    echo "📊 First run — generating dashboard data..."
    echo "   (This fetches stock data & runs backtests. Takes ~2 min)"
    echo ""
    python dashboard/prepare_dash_data.py
    if [ $? -ne 0 ]; then
        echo "❌ Data generation failed!"
        exit 1
    fi
else
    echo "✅ Dashboard data ready"
fi

# Launch dashboard & open browser
echo ""
echo "🌐 Launching dashboard at http://localhost:8050"
echo "   Press Ctrl+C to stop"
echo "============================================"
echo ""

# Open browser after a short delay (runs in background)
(sleep 2 && open "http://localhost:8050") &

# Start Dash server
python dashboard/dash_app.py
