#!/bin/bash

# Boot Selector Script
# Allows user to select OS for next boot only, then returns to default

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo -e "${RED}Error: This script should not be run as root${NC}"
   echo "Run it as a regular user - it will prompt for sudo when needed"
   exit 1
fi

echo -e "${BLUE}=== Boot Selector - Next Boot Only ===${NC}"
echo "This will set the OS for the NEXT boot only, then return to default (Linux)"
echo

# Check if grub-reboot is available
if ! command -v grub-reboot &> /dev/null; then
    echo -e "${RED}Error: grub-reboot command not found${NC}"
    echo "Make sure GRUB is properly installed"
    exit 1
fi

# Function to extract menu entries from GRUB config
get_boot_entries() {
    local grub_cfg="/boot/grub/grub.cfg"
    
    if [[ ! -f "$grub_cfg" ]]; then
        echo -e "${RED}Error: GRUB config file not found at $grub_cfg${NC}"
        exit 1
    fi
    
    # Get all menuentry lines without using pipe in function
    local temp_file=$(mktemp)
    sudo grep "menuentry '" "$grub_cfg" | grep -v "menuentry_id_option=" > "$temp_file"
    
    while IFS= read -r line_content; do
        # Extract the title between single quotes
        if [[ "$line_content" =~ menuentry\ \'([^\']+)\' ]]; then
            title="${BASH_REMATCH[1]}"
            
            # Skip advanced options submenu (we'll show individual entries)
            if [[ "$title" =~ "Advanced options" ]]; then
                continue
            fi
            
            # Extract entry ID - try different patterns
            entry_id=""
            if [[ "$line_content" =~ \$menuentry_id_option\ \'([^\']+)\' ]]; then
                entry_id="${BASH_REMATCH[1]}"
            elif [[ "$line_content" =~ --id\ \'([^\']+)\' ]]; then
                entry_id="${BASH_REMATCH[1]}"
            else
                # Fallback: use title as ID
                entry_id="$title"
            fi
            
            echo "$entry_id|$title"
        fi
    done < "$temp_file"
    
    rm -f "$temp_file"
}

# Get available boot entries
echo -e "${YELLOW}Scanning available boot options...${NC}"
echo

declare -a entries
declare -a entry_ids
index=0

# Get entries and read into arrays using process substitution
while IFS='|' read -r entry_id title; do
    entries[$index]="$title"
    entry_ids[$index]="$entry_id"
    ((index++))
done < <(get_boot_entries)

if [[ ${#entries[@]} -eq 0 ]]; then
    echo -e "${RED}Error: No boot entries found${NC}"
    exit 1
fi

# Display menu
echo -e "${GREEN}Available Boot Options:${NC}"
echo
for i in "${!entries[@]}"; do
    echo "  $((i+1)). ${entries[$i]}"
done
echo

# Get user selection
while true; do
    read -p "Select boot option (1-${#entries[@]}) or 'q' to quit: " choice
    
    if [[ "$choice" == "q" ]] || [[ "$choice" == "Q" ]]; then
        echo "Cancelled."
        exit 0
    fi
    
    if [[ "$choice" =~ ^[0-9]+$ ]] && [[ "$choice" -ge 1 ]] && [[ "$choice" -le ${#entries[@]} ]]; then
        selected_index=$((choice-1))
        selected_entry="${entries[$selected_index]}"
        selected_id="${entry_ids[$selected_index]}"
        break
    else
        echo -e "${RED}Invalid selection. Please enter a number between 1 and ${#entries[@]}${NC}"
    fi
done

echo
echo -e "${YELLOW}Selected: $selected_entry${NC}"
echo -e "${BLUE}This will be used for the NEXT boot only, then return to default (Linux)${NC}"
echo

# Confirm selection
read -p "Proceed with this selection? (y/N): " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Set the boot entry for next boot only
echo
echo -e "${YELLOW}Setting boot option for next boot...${NC}"

# Try different approaches to set the boot entry
if sudo grub-reboot "$selected_id" 2>/dev/null; then
    echo -e "${GREEN}✓ Boot option set successfully${NC}"
elif sudo grub-reboot "$selected_entry" 2>/dev/null; then
    echo -e "${GREEN}✓ Boot option set successfully${NC}"
elif sudo grub-reboot "$((selected_index))" 2>/dev/null; then
    echo -e "${GREEN}✓ Boot option set successfully${NC}"
else
    echo -e "${RED}Error: Failed to set boot option${NC}"
    echo "You may need to manually select the OS from GRUB menu during boot"
    exit 1
fi

echo
echo -e "${GREEN}=== Setup Complete ===${NC}"
echo -e "${BLUE}Next boot will use: $selected_entry${NC}"
echo -e "${BLUE}After that boot, the system will return to default (Linux)${NC}"
echo

# Ask if user wants to reboot now
read -p "Reboot now? (y/N): " reboot_now
if [[ "$reboot_now" =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Rebooting in 3 seconds...${NC}"
    sleep 3
    sudo reboot
else
    echo -e "${GREEN}System ready. Reboot when you're ready to use the selected OS.${NC}"
fi