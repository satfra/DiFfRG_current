#!/bin/bash

echo "###############################################################"
echo "## DiFfRG Docker Build Script"
echo "###############################################################"
echo
echo "Choose a setup to build:"
echo "1) Base"
echo "2) MPI+OpenMP"
echo "3) MPI+OpenMP+CUDA"
echo
echo "Enter your choice (1-3):"
read choice
echo
# Check if the choice is valid
if [[ "$choice" -lt 1 || "$choice" -gt 3 ]]; then
    echo "Invalid choice. Please enter a number between 1 and 3."
    exit 1
fi
# Get the list of images based on the choice
if [[ "$choice" -eq 1 ]]; then
    images=$(ls Base/)
    folder="Base"
elif [[ "$choice" -eq 2 ]]; then
    images=$(ls MPI+OpenMP/)
    folder="MPI+OpenMP"
else
    images=$(ls MPI+OpenMP+CUDA/)
    folder="MPI+OpenMP+CUDA"
fi
echo "The following images are available:"
# enumerate the images
i=1
for image in $images; do
    echo "$i) $image"
    ((i++))
done
echo
echo "Enter the number of the image you want to build:"
read image_choice
# Check if the choice is valid
if [[ "$image_choice" -lt 1 || "$image_choice" -gt $(echo "$images" | wc -l) ]]; then
    echo "Invalid choice. Please enter a number between 1 and $(echo "$images" | wc -l)."
    exit 1
fi
# Get the selected image
image=$(echo "$images" | sed -n "${image_choice}p")

echo "   Building DiFfRG docker image diffrg-$image..."
docker buildx build -t diffrg-$image -f ${folder}/${image} . --no-cache --progress=plain &>logs/$image.log
if [ $? -ne 0 ]; then
    echo "   Error building ${folder}/${image}. Check the log file logs/${image}.log for details."
else
    echo "   Successfully built ${folder}/${image}."
fi
