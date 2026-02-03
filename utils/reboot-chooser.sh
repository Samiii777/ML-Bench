#!/bin/bash

# Script to list installed OSes and reboot into a selected one
# Uses GRUB bootloader configuration

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# GRUB config file location
GRUB_CFG="/boot/grub/grub.cfg"

# Show all entries flag
SHOW_ALL=0

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        echo -e "${RED}Error: This script must be run as root (use sudo)${NC}"
        exit 1
    fi
}

# Check if GRUB config exists
check_grub() {
    if [[ ! -f "$GRUB_CFG" ]]; then
        # Try alternative location
        GRUB_CFG="/boot/grub2/grub.cfg"
        if [[ ! -f "$GRUB_CFG" ]]; then
            echo -e "${RED}Error: GRUB configuration file not found${NC}"
            echo "Checked: /boot/grub/grub.cfg and /boot/grub2/grub.cfg"
            exit 1
        fi
    fi
}

# Parse GRUB config and extract menu entries
get_all_boot_entries() {
    # Use sed to extract menuentry names
    # Handles both 'single quoted' and "double quoted" names
    sed -n "s/^[[:space:]]*menuentry '\([^']*\)'.*/\1/p; s/^[[:space:]]*menuentry \"\([^\"]*\)\".*/\1/p" "$GRUB_CFG"
}

# Filter entries to show only relevant OSes (Windows + other Linux installations)
get_filtered_boot_entries() {
    get_all_boot_entries | while read -r entry; do
        # Include Windows entries
        if [[ "$entry" == *[Ww]indows* ]]; then
            echo "$entry"
        # Include other OS installations (entries with "on /dev/" indicating different drive/partition)
        # But exclude recovery modes and specific kernel versions from other installs
        elif [[ "$entry" == *"(on /dev/"* ]]; then
            # Skip recovery modes
            [[ "$entry" == *"recovery mode"* ]] && continue
            # Skip entries that are just kernel versions (contain "with Linux")
            [[ "$entry" == *"with Linux"* ]] && continue
            echo "$entry"
        fi
    done
}

# Get boot entries based on SHOW_ALL flag
get_boot_entries() {
    if [[ "$SHOW_ALL" -eq 1 ]]; then
        get_all_boot_entries
    else
        get_filtered_boot_entries
    fi
}

# Display menu and get user choice
display_menu() {
    local entries=("$@")
    local count=${#entries[@]}
    
    echo -e "\n${CYAN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║${NC}             ${GREEN}Available Operating Systems${NC}                      ${CYAN}║${NC}"
    echo -e "${CYAN}╚════════════════════════════════════════════════════════════╝${NC}\n"
    
    for i in "${!entries[@]}"; do
        local entry="${entries[$i]}"
        # Clean up the display name
        local display_name="${entry//>/  →  }"
        printf "${YELLOW}  [%2d]${NC} %s\n" "$((i+1))" "$display_name"
    done
    
    echo -e "\n${CYAN}────────────────────────────────────────────────────────────────${NC}"
    echo -e "  ${BLUE}[0]${NC}  Cancel and exit"
    echo -e "${CYAN}────────────────────────────────────────────────────────────────${NC}"
    if [[ "$SHOW_ALL" -eq 0 ]]; then
        echo -e "  ${BLUE}Tip:${NC} Use ${YELLOW}--all${NC} flag to see all boot entries\n"
    else
        echo ""
    fi
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            -a|--all)
                SHOW_ALL=1
                shift
                ;;
            -h|--help)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  -a, --all    Show all boot entries (including kernels, recovery, etc.)"
                echo "  -h, --help   Show this help message"
                exit 0
                ;;
            *)
                echo "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
}

# Main function
main() {
    parse_args "$@"
    
    echo -e "\n${GREEN}=== OS Reboot Chooser ===${NC}\n"
    
    # Checks
    check_root
    check_grub
    
    echo -e "${BLUE}Reading GRUB configuration from: ${NC}$GRUB_CFG"
    
    # Debug: show raw menuentry lines if DEBUG is set
    if [[ "${DEBUG:-}" == "1" ]]; then
        echo -e "\n${YELLOW}DEBUG: Raw menuentry lines found:${NC}"
        grep -E "menuentry\s+" "$GRUB_CFG" | head -20
        echo ""
    fi
    
    # Get boot entries into an array
    mapfile -t boot_entries < <(get_boot_entries)
    
    if [[ ${#boot_entries[@]} -eq 0 ]]; then
        echo -e "${RED}No boot entries found in GRUB configuration${NC}"
        exit 1
    fi
    
    # Display menu
    display_menu "${boot_entries[@]}"
    
    # Get user choice
    local choice
    while true; do
        read -rp "$(echo -e ${GREEN}Enter your choice [0-${#boot_entries[@]}]: ${NC})" choice
        
        # Validate input
        if [[ "$choice" =~ ^[0-9]+$ ]]; then
            if [[ "$choice" -eq 0 ]]; then
                echo -e "\n${YELLOW}Cancelled. No changes made.${NC}\n"
                exit 0
            elif [[ "$choice" -ge 1 && "$choice" -le ${#boot_entries[@]} ]]; then
                break
            fi
        fi
        echo -e "${RED}Invalid choice. Please enter a number between 0 and ${#boot_entries[@]}${NC}"
    done
    
    # Get the selected entry (0-indexed)
    local selected_index=$((choice - 1))
    local selected_entry="${boot_entries[$selected_index]}"
    
    echo -e "\n${BLUE}Selected: ${NC}${selected_entry//>/  →  }"
    
    # Confirm reboot
    echo ""
    read -rp "$(echo -e ${YELLOW}Do you want to reboot into this OS now? [y/N]: ${NC})" confirm
    
    if [[ "${confirm,,}" == "y" || "${confirm,,}" == "yes" ]]; then
        echo -e "\n${BLUE}Setting next boot entry...${NC}"
        
        # Use grub-reboot to set next boot entry
        # grub-reboot uses the entry string format "submenu>entry" for nested entries
        if grub-reboot "$selected_entry" 2>/dev/null || grub2-reboot "$selected_entry" 2>/dev/null; then
            echo -e "${GREEN}Successfully set next boot to: ${NC}${selected_entry//>/  →  }"
            echo -e "\n${YELLOW}Rebooting in 3 seconds...${NC}"
            sleep 3
            reboot
        else
            echo -e "${RED}Failed to set boot entry. Trying alternative method...${NC}"
            # Try with index instead
            if grub-reboot "$selected_index" 2>/dev/null || grub2-reboot "$selected_index" 2>/dev/null; then
                echo -e "${GREEN}Successfully set next boot entry${NC}"
                echo -e "\n${YELLOW}Rebooting in 3 seconds...${NC}"
                sleep 3
                reboot
            else
                echo -e "${RED}Failed to set boot entry. Please check your GRUB configuration.${NC}"
                exit 1
            fi
        fi
    else
        echo -e "\n${YELLOW}Reboot cancelled. The boot entry was NOT changed.${NC}"
        echo -e "${BLUE}To manually set and reboot later, run:${NC}"
        echo -e "  sudo grub-reboot \"$selected_entry\" && sudo reboot\n"
    fi
}

# Run main function
main "$@"
