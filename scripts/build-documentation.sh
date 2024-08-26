#!/bin/bash

doc_dir="${DOC_DIR:-.cmmvae/docs}"

replace_placeholders() {
  local root_dir=$1

  if [ -z "$GIT_REPO_OWNER" ] || [ -z "$GIT_REPO_NAME" ]; then
    echo "Error: GIT_REPO_OWNER and GIT_REPO_NAME environment variables must be set."
    exit 1
  fi

  find "$root_dir" -type f | while read -r file; do
    # Replace placeholders
    sed -i "s|&lt;GIT_REPO_OWNER&gt;|$GIT_REPO_OWNER|g" "$file"
    sed -i "s|&lt;GIT_REPO_NAME&gt;|$GIT_REPO_NAME|g" "$file"
  done
}

script_dir=$(dirname "$0")
# Run the pdoc command with the specified or default output directory
"$script_dir/pdoc.sh" --output "$doc_dir"

# Replace placeholders in the generated documentation
replace_placeholders "$doc_dir"
