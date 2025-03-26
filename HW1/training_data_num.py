import os
import matplotlib.pyplot as plt

# Set the path to your train folder
train_dir = "./data/train"

# Dictionary to store image counts
image_counts = {}

# Loop through each class folder (0-99)
for class_name in sorted(os.listdir(train_dir), key=lambda x: int(x)):  
    class_path = os.path.join(train_dir, class_name)
    if os.path.isdir(class_path):
        num_images = len([f for f in os.listdir(class_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
        image_counts[class_name] = num_images

# Find the class with the min and max images
min_class = min(image_counts, key=image_counts.get)
max_class = max(image_counts, key=image_counts.get)

print(f"Class with the fewest images: {min_class} ({image_counts[min_class]} images)")
print(f"Class with the most images: {max_class} ({image_counts[max_class]} images)")

# Plot the image counts
plt.figure(figsize=(15, 6))
bars = plt.bar(image_counts.keys(), image_counts.values(), color='skyblue', width=0.6)  # 調整 width 讓長條之間有更多間隔

# Highlight min and max with red and green
bars[int(min_class)].set_color('red')  # Min in red
bars[int(max_class)].set_color('green')  # Max in green

# Labels and title
plt.xlabel("Class (0-99)")
plt.ylabel("Number of Images")
plt.title("Number of Images in Each Class")
plt.xticks(range(0, 100, 5))  # 每 5 個類別顯示一個標籤，減少擁擠
plt.grid(axis='y', linestyle='--', alpha=0.7)  # 增加背景虛線網格，讓長條更清楚
plt.legend(["Classes", f"Min: {min_class} ({image_counts[min_class]})", f"Max: {max_class} ({image_counts[max_class]})"])

# Save the plot
plt.savefig("image_distribution.png", dpi=300, bbox_inches='tight')
print("Saved as image_distribution.png")
