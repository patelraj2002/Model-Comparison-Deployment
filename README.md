1. Objective
This project directly addresses by performing a comparative analysis of two object
detection models: YOLOv5n (nano) and YOLOv5s (small). The evaluation is conducted on a
consistent set of 10 images to measure key performance indicators:
● Inference Speed: Average time taken to process an image.
● Detection Quantity: The total number of objects detected.
● Class Diversity: The range of unique object classes identified.
A critical requirement of this task was deployment readiness. To meet this, the entire application
has been containerized using Docker. This approach encapsulates all dependencies and code
into a portable, reproducible environment that can be executed with a single command,
demonstrating a best-practice MLOps workflow.
2. Methodology & Project Design
The project is structured as a self-contained application, designed for execution within a Docker
container. This ensures that the runtime environment is identical regardless of the host machine.
Project Workflow
1. Input: The application accepts a folder of images (input_image/) as its primary input.
2. Processing: A Python script (model_comparison.py) orchestrates the entire process. It
loads both YOLOv5 models, iterates through each input image, and runs inference with
both models.
3. Analysis: For each image, the script records inference time, counts the number of
detections, and identifies the unique classes found. These metrics are then aggregated
to produce a final comparison report.
4. Output: The application generates two types of output into a specified directory
(result/):
○ Annotated Images: Visual copies of the input images with bounding boxes
drawn for both models.
○ CSV Reports: Detailed per-image metrics and an aggregate summary table
comparing the two models.
5.
The full implementation details, including the Python script, Dockerfile, and dependency list
(requirements.txt), are available in the accompanying submission folder and Jupyter
Notebook.
3. How to Run the Project
Prerequisites: Docker Desktop must be installed and running.
Step 1: Build the Docker Image
Open a terminal, navigate to the project's root directory (Task_3_Project/), and execute the
following command. This will build a Docker image named model-comparison-app containing
the application and all its dependencies.
docker build -t model-comparison-app .
Step 2: Run the Analysis
Execute the command below to start the container. This command links the local input_image
and result folders to the container, allowing it to process local files and save the output back to
the host machine
docker run --rm -v "$(pwd)/input_image:/app/input" -v "$(pwd)/result:/app/output"
model-comparison-app
Upon completion, all generated reports and annotated images will be available in the result
folder on the local machine.
4. Results and Conclusion
The following is a summary of the output generated from a sample run.
4.1. Quantitative Comparison
Metric YOLOv5n
(Nano)
YOLOv5s
(Small)
Performance Ratio
(s/n)
Average Inference
Time (s)
0.297 0.278 0.94x
Average Detection
Count
28.0 37.0 1.32x
Average Class
Diversity
5.0 5.0 1.00x
4.2. Analysis
The results highlight the classic trade-off between model complexity and performance:
● Higher Detection Rate: The larger YOLOv5s model detected 32% more objects than
the lightweight YOLOv5n model, demonstrating superior recall and sensitivity.
● Comparable Speed: Despite being a larger model, YOLOv5s performed with
comparable, and in this test, slightly faster, inference speed on a CPU.
● Conclusion: For applications where maximizing detection accuracy is the primary goal,
YOLOv5s offers a significant performance advantage with a negligible impact on
speed. The successful containerization of this workflow confirms its readiness for
scalable and reliable deployment.
