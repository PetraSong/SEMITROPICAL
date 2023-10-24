#!/bin/bash

# author: Sander W. van der Laan | s.w.vanderlaan-2@umcutrecht.nl
# last update: 2023-10-23

### Creating display functions
### Setting colouring
NONE='\033[00m'
BOLD='\033[1m'
# OPAQUE='\033[2m'
FLASHING='\033[5m'
# UNDERLINE='\033[4m'

RED='\033[01;31m'
# GREEN='\033[01;32m'
# YELLOW='\033[01;33m'
# PURPLE='\033[01;35m'
CYAN='\033[01;36m'
# WHITE='\033[01;37m'
### Regarding changing the 'type' of the things printed with 'echo'
### Refer to: 
### - http://askubuntu.com/questions/528928/how-to-do-underline-bold-italic-strikethrough-color-background-and-size-i
### - http://misc.flogisoft.com/bash/tip_colors_and_formatting
### - http://unix.stackexchange.com/questions/37260/change-font-in-echo-command

### echo -e "\033[1mbold\033[0m"
### echo -e "\033[3mitalic\033[0m" ### THIS DOESN'T WORK ON MAC!
### echo -e "\033[4munderline\033[0m"
### echo -e "\033[9mstrikethrough\033[0m"
### echo -e "\033[31mHello World\033[0m"
### echo -e "\x1B[31mHello World\033[0m"

# for i in $(seq 0 5) 7 8 $(seq 30 37) $(seq 41 47) $(seq 90 97) $(seq 100 107) ; do 
# 	echo -e "\033["$i"mYou can change the font...\033[0m"; 
# done
### Creating some function
# function echobold { #'echobold' is the function name
#     echo -e "\033[1m${1}\033[0m" # this is whatever the function needs to execute.
# }
function echocyan { #'echobold' is the function name
    echo -e "${CYAN}${1}${NONE}" # this is whatever the function needs to execute.
}
function echobold { #'echobold' is the function name
    echo -e "${BOLD}${1}${NONE}" # this is whatever the function needs to execute, note ${1} is the text for echo
}
function echoitalic { #'echobold' is the function name
    echo -e "\033[3m${1}\033[0m" # this is whatever the function needs to execute.
}
function echowarning { #'echobold' is the function name
    echo -e "${FLASHING}${RED}${1}${NONE}" # this is whatever the function needs to execute.
}

help_function() {
  echocyan "Usage: $0 --folder folder_path --step [segm|feat]"
  echocyan "--folder_path: should be /path/to/your/folder/STAIN/_images"
  echocyan "--step: either 'segm' or 'feat'"
  echocyan "--debug: enable debug mode (optional)"
  echocyan ""
  echocyan "Usage: $0 --help"
  echocyan "--help: for this help"
  echocyan ""
  echocyan "Description: This script will check whether the segmentation or feature extraction were successful and lists any samples that have missing information."
  echocyan ""
}

# Check for the --help argument
if [ "$1" == "--help" ]; then
  help_function
  exit 0
fi

# Check for the correct number of arguments
if [ "$#" -lt 4 ] || [ "$#" -gt 5 ]; then
  echowarning "Invalid number of arguments."
  help_function
  exit 1
fi

debug=false  # Default value for debug

# Parse command-line arguments
while [ "$#" -gt 0 ]; do
  case "$1" in
    --folder)
      folder_path="$2"
      shift 2
      ;;
    --step)
      step="$2"
      shift 2
      ;;
    --debug)
      debug=true
      shift
      ;;
    *)
      echowarning "Invalid argument: $1"
      help_function
      exit 1
      ;;
  esac
done

# Initialize a counter for progress
progress_counter=0

# Function to print a progress bar
print_progress() {
    local progress="$1"  # Pass the progress value as an argument
    local bar_length=20  # Adjust the length of the progress bar

    # Ensure the progress value is within the range [0, 100]
    progress=$((progress < 0 ? 0 : progress))
    progress=$((progress > 100 ? 100 : progress))

    # Calculate the number of bars to display
    local num_bars=$((progress * bar_length / 100))
    
    # Create the progress bar with dots
    local bar=""
    for ((i=0; i<num_bars; i++)); do
        bar+="="
    done

    # Calculate the number of spaces
    local num_spaces=$((bar_length - num_bars))
    local space=""
    for ((i=0; i<num_spaces; i++)); do
        space+=" "
    done

    # Print the progress bar
    echo -ne "Progress: [$bar$space] $progress%\r"
}


echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
echobold "              CHECK PROCESSING OF SEGMENTATION AND FEATURE EXTRACTION"
echoitalic "                                    version 1.0"
echo ""
echoitalic "* Written by  : Sander W. van der Laan"
echoitalic "* E-mail      : s.w.vanderlaan-2@umcutrecht.nl"
echoitalic "* Last update : 2023-10-23"
echoitalic "* Version     : v1.0"
echo ""
echoitalic "* Description : This script will check whether the segmentation or feature "
echoitalic "                extraction were succesful and lists any samples that has "
echoitalic "                missing information."
echoitalic ""
echoitalic "                Use \`bash check.segm_feat.sh --help\` to get options."
echo ""
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
echo "Today's: "$(date)
TODAY=$(date +"%Y%m%d")
echo ""
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"

echobold "Processing folder [$folder_path] on $TODAY."

if [ "$debug" = true ]; then
  echocyan "Debug mode is enabled."
fi

if [ "$step" == "segm" ]; then
  # Only check two folders for 'segm' step
  search_folders=(
  "$folder_path/PROCESSED/patches_512"
  "$folder_path/PROCESSED/masks/_images"
  )
elif [ "$step" == "feat" ]; then
  # Check the other folders for 'feat' step
  search_folders=(
  "$folder_path/PROCESSED/thumbnails"
  "$folder_path/PROCESSED/features/h5_files"
  "$folder_path/PROCESSED/features/pt_files"
  # Add more folders as needed
  )
else
  echowarning "Invalid 'step' argument. Use 'segm' or 'feat'."
  help_function
  exit 1
fi

# Use 'ls' to list the files in the folder and 'grep' to extract the desired pattern
# In this case, it will extract strings that start with 'AE' followed by numbers.
first_part=$(ls "$folder_path" | grep -oE 'AE[0-9]+')
if [ "$debug" = true ]; then
  echocyan "Checking files for studynumbers: \n$first_part"  # Debug line to display the samples to check
fi

# Iterate through each file in the source folder (first_part) and check if it exists in the search folders (search_folders)
for file in $first_part; do
  found=false
  if [ "$debug" = true ]; then
    echocyan "Checking $file in search folders."  # Debug line to display the current search folder
  fi

  for search_folder in "${search_folders[@]}"; do
    # Use the 'find' command to search for files that match the pattern "$file.*" in the current search folder
    # if [[ -n $(find "$search_folder" -type f -name "$file.*") ]]; then
    if [[ -n $(find "$search_folder" -type f -name "$file*") ]]; then
    # if ! find "$search_folder" -type f -name "$file*" >/dev/null; then
      if [ "$debug" = true ]; then
        echocyan "Found files matching '$file*' in '$search_folder'."
      fi
      found=true
    fi
  done

  if [ "$found" = false ]; then
    echo "WARNING: No files matching '$file.*' found in any search folder."
  fi
  
  # Increment the progress counter
  ((progress_counter += 1))
  # Calculate the progress percentage
  progress_percentage=$((progress_counter * 100 / ${#first_part}))

  # Update and print the progress bar
  print_progress $progress_percentage
done

# Add this line to display 100% after processing
print_progress 100

echo ""
echo "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
echobold "Wow! That was a lot of work. All checks are complete. Let's have a beer, buddy!"
date