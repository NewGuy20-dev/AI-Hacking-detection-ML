# Host-Based Detection Datasets - Download Status

**Last Updated:** December 22, 2025

---

## Download Progress

| Dataset | Size | Status | Location | Notes |
|---------|------|--------|----------|-------|
| ADFA-IDS | 500 MB | ✅ Auto-downloaded | `datasets/adfa_ids/` | System call sequences (Linux/Windows) |
| EVTX-ATTACK-SAMPLES | 300 MB | ✅ Auto-downloaded | `datasets/evtx_samples/` | Windows event logs (270+ samples) |
| DARPA | 221 MB | ⚠️ Manual | `datasets/darpa/` | File system dumps + audit logs |
| CIC MalMem-2022 | 1.2 GB | ⚠️ Manual | `datasets/cic_malmem/` | Memory dump analysis (58,596 samples) |

---

## Auto-Downloaded Datasets ✅

### 1. ADFA-IDS (500 MB)
```
Location: datasets/adfa_ids/
Status: ✅ COMPLETE
Contents:
  - ADFA-LD/ (Linux system calls)
  - ADFA-WD/ (Windows system calls)
  - README files
```

### 2. EVTX-ATTACK-SAMPLES (300 MB)
```
Location: datasets/evtx_samples/
Status: ✅ COMPLETE
Contents:
  - Privilege Escalation/
  - Lateral Movement/
  - Persistence/
  - Credential Access/
  - 270+ EVTX samples mapped to MITRE ATT&CK
```

---

## Manual Download Required ⚠️

### 3. DARPA (221 MB)
**Download from:** https://www.ll.mit.edu/r-d/datasets/1998-darpa-intrusion-detection-evaluation-dataset

**Steps:**
1. Visit the link above
2. Download 1998 DARPA dataset
3. Extract to `datasets/darpa/`

**Contents:**
- File system dumps (ufsdump)
- Audit logs
- Network traffic (tcpdump)

---

### 4. CIC MalMem-2022 (1.2 GB)
**Download from:** https://www.unb.ca/cic/datasets/malmem-2022.html

**Steps:**
1. Visit the link above
2. Register/login if required
3. Download CIC-MalMem-2022 dataset
4. Extract to `datasets/cic_malmem/`

**Contents:**
- 58,596 memory dump samples
- 29,298 benign + 29,298 malicious
- Spyware, Ransomware, Trojan Horse families

---

## Total Dataset Size

| Category | Size |
|----------|------|
| Auto-downloaded | 800 MB |
| Manual download | 1.4 GB |
| **Total** | **~2.2 GB** |

---

## Next Steps

1. ✅ Download ADFA-IDS and EVTX-ATTACK-SAMPLES (auto-complete)
2. ⏳ Manually download DARPA dataset
3. ⏳ Manually download CIC MalMem-2022 dataset
4. ⏳ Verify all datasets in `datasets/` directory
5. ⏳ Run preprocessing scripts
6. ⏳ Train host-based detection models
7. ⏳ Integrate with ensemble voting

---

## Verification Commands

```bash
# Check downloaded datasets
ls -lh datasets/adfa_ids/
ls -lh datasets/evtx_samples/

# Check total size
du -sh datasets/

# List all dataset directories
ls -d datasets/*/
```

---

**Status:** 2/4 datasets auto-downloaded, 2/4 require manual download
**Total Progress:** 50% complete
