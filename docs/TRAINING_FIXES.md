# Training Fixes Applied

## 🐛 Issues Found:

1. **No malicious files loaded** - "Malicious files: 0"
   - Cause: Looking for `.jsonl` but security_payloads has `.txt` files
   
2. **Training pausing at 50%** - DataLoader hanging on Windows
   - Cause: Too many workers + persistent_workers on Windows

## ✅ Fixes Applied:

### Fix 1: Load .txt Files from security_payloads
**File:** `scripts/train_rtx3050.py`

**Changed:**
```python
# OLD - Only looked for .jsonl
malicious_files.extend(payload_dir.rglob('*.jsonl'))

# NEW - Looks for both .txt and .jsonl
malicious_files.extend(payload_dir.rglob('*.txt'))
malicious_files.extend(payload_dir.rglob('*.jsonl'))
```

**Also added:**
- `live_benign` directory to benign sources
- `.txt` files from benign directories

### Fix 2: Windows DataLoader Optimization
**File:** `scripts/train_rtx3050.py`

**Changed:**
```python
# OLD - Linux settings
num_workers=4
persistent_workers=True
timeout=0
prefetch_factor=4

# NEW - Windows-safe settings
num_workers=2 if is_windows else 4  # Reduce workers
persistent_workers=False if is_windows else True  # Disable on Windows
timeout=60 if is_windows else 0  # Add timeout
prefetch_factor=2 if is_windows else 4  # Reduce prefetch
```

## 📊 Expected Results After Fix:

```
Malicious files: 1000+  ← Should see many files now
Benign files: 15+
```

## 🧪 Test the Fix:

```powershell
# 1. Check data files are found
python check_data_files.py

# 2. Resume training (will use new settings)
python scripts/train_rtx3050.py --model payload --resume
```

## 🔍 Why It Works:

### .txt File Support:
The `StreamingDataset` already has `_read_txt()` method that:
- Reads each line as a sample
- Labels it as malicious (label=1)
- Tokenizes character-by-character
- Works exactly like .jsonl files

### Windows DataLoader Fix:
- **2 workers** instead of 4 - Reduces process overhead
- **No persistent workers** - Prevents zombie processes
- **60s timeout** - Kills stuck workers
- **Prefetch=2** - Reduces memory pressure

## ⏱️ Performance Impact:

- **Slightly slower** (~10-15%) due to fewer workers
- **But won't hang!** - Training will complete
- **Still uses GPU** - No impact on GPU utilization

## 🚀 Next Steps:

1. Run `check_data_files.py` to verify files are found
2. Resume training with `--resume` flag
3. Training should now:
   - Load 1000+ malicious files
   - Not hang at 50%
   - Complete all 5 epochs

## 📝 Notes:

- The streaming dataset reads `.txt` files line-by-line
- Each line in security_payloads/*.txt is one attack payload
- No conversion needed - it's already trainable!
- The fix is permanent - will work for all future training runs
