# /bin/sh
rm -r *.pt runs up 
find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf
find . -name '.DS_Store' -type f -ls -delete
#find . -name "*.swp" -name "*.swm" -name "*.swo" -name "*.swn" -type f -ls -delete
