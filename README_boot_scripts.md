# Boot Selector Scripts

This directory contains scripts to temporarily boot into different operating systems and automatically return to Linux after the next reboot.

## Scripts

### 1. `quick_windows_boot.sh` (Recommended)
**Simple script specifically for Windows/Linux dual-boot setups**

```bash
./quick_windows_boot.sh
```

**Features:**
- Automatically detects Windows Boot Manager
- Sets Windows for next boot only
- After Windows reboot, automatically returns to Linux
- User-friendly interface with confirmation prompts
- Optional immediate reboot

**Use case:** Perfect for when you need to quickly boot into Windows and want to automatically return to Linux afterward.

### 2. `boot_selector.sh` (Advanced)
**Full-featured script that shows all available boot options**

```bash
./boot_selector.sh
```

**Features:**
- Lists all available boot entries (Ubuntu kernels, Windows, etc.)
- Interactive menu to select any boot option
- Sets selected OS for next boot only
- Returns to default (Linux) after that boot
- Handles multiple Ubuntu kernel versions

**Use case:** When you need to boot into recovery mode, different kernel versions, or want to see all available options.

## How It Works

Both scripts use GRUB's `grub-reboot` command, which:
- Sets the boot entry for the **next boot only**
- After that boot completes and the system reboots again, it returns to the default OS (Linux)
- This is perfect for temporary OS switches

## System Requirements

- Linux system with GRUB bootloader
- Dual-boot or multi-boot setup
- sudo privileges
- `grub-reboot` command available

## Example Usage Scenario

**Goal:** Boot into Windows temporarily, then return to Linux automatically

1. Run the script:
   ```bash
   ./quick_windows_boot.sh
   ```

2. Script detects: "Windows Boot Manager (on /dev/nvme0n1p1)"

3. Confirm and reboot into Windows

4. Do your work in Windows

5. When you reboot from Windows, you automatically return to Linux (no manual selection needed)

## Safety Features

- Scripts require user confirmation before making changes
- Only affects the next boot (not permanent)
- Will not run as root (for safety)
- Validates GRUB entries before proceeding
- Provides clear feedback on success/failure

## Troubleshooting

If the scripts don't work:

1. **Check if Windows is detected:**
   ```bash
   sudo grep -i "windows" /boot/grub/grub.cfg
   ```

2. **Update GRUB if needed:**
   ```bash
   sudo update-grub
   ```

3. **Check available entries manually:**
   ```bash
   sudo grep "menuentry" /boot/grub/grub.cfg
   ```

## Current System Status

Based on your system scan:
- **Current OS:** Ubuntu 24.04.2 LTS 
- **Windows Available:** Yes (Windows Boot Manager detected)
- **GRUB Configuration:** `/boot/grub/grub.cfg`
- **Default Boot:** Linux (GRUB_DEFAULT=0)

The scripts are ready to use on your system!