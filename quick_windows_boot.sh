#!/bin/bash

# Quick Windows Boot Script
# Simple script to boot into Windows for next boot only, then return to Linux

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo -e "${RED}Error: This script should not be run as root${NC}"
   echo "Run it as a regular user - it will prompt for sudo when needed"
   exit 1
fi

echo -e "${BLUE}=== Quick Windows Boot ===${NC}"
echo "This will boot into Windows for the NEXT boot only"
echo "After Windows reboot, you'll automatically return to Linux"
echo

# Find Windows boot entry
windows_entry=$(sudo grep -i "windows boot manager\|microsoft" /boot/grub/grub.cfg | grep menuentry | head -1)

if [[ -z "$windows_entry" ]]; then
    echo -e "${RED}Error: Windows boot entry not found${NC}"
    echo "Windows may not be installed or not detected by GRUB"
    exit 1
fi

# Extract Windows entry details
if [[ "$windows_entry" =~ menuentry\ \'([^\']+)\' ]]; then
    windows_title="${BASH_REMATCH[1]}"
    echo -e "${GREEN}Found Windows: $windows_title${NC}"
else
    echo -e "${RED}Error: Could not parse Windows entry${NC}"
    exit 1
fi

# Get the entry ID for Windows
if [[ "$windows_entry" =~ \$menuentry_id_option\ \'([^\']+)\' ]]; then
    windows_id="${BASH_REMATCH[1]}"
elif [[ "$windows_entry" =~ --id\ \'([^\']+)\' ]]; then
    windows_id="${BASH_REMATCH[1]}"
else
    # Fallback to using the title
    windows_id="$windows_title"
fi

echo
echo -e "${YELLOW}Current OS: $(cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)${NC}"
echo -e "${YELLOW}Next boot will use: $windows_title${NC}"
echo -e "${BLUE}After Windows reboot, you'll return to Linux automatically${NC}"
echo

# Confirm
read -p "Boot into Windows on next reboot? (y/N): " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Set Windows for next boot
echo
echo -e "${YELLOW}Setting Windows for next boot...${NC}"

if sudo grub-reboot "$windows_id" 2>/dev/null; then
    echo -e "${GREEN}✓ Windows boot set successfully${NC}"
elif sudo grub-reboot "$windows_title" 2>/dev/null; then
    echo -e "${GREEN}✓ Windows boot set successfully${NC}"
else
    echo -e "${RED}Error: Failed to set Windows boot${NC}"
    echo "You may need to manually select Windows from GRUB menu"
    exit 1
fi

echo
echo -e "${GREEN}=== Ready ===${NC}"
echo -e "${BLUE}✓ Next boot: Windows${NC}"
echo -e "${BLUE}✓ After Windows reboot: Back to Linux (automatic)${NC}"
echo

# Ask if user wants to reboot now
read -p "Reboot now? (y/N): " reboot_now
if [[ "$reboot_now" =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Rebooting to Windows in 3 seconds...${NC}"
    sleep 3
    sudo reboot
else
    echo -e "${GREEN}System ready. Reboot when you want to use Windows.${NC}"
    echo -e "${BLUE}Remember: This is only for the next boot!${NC}"
fi