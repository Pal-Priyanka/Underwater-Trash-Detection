# 🛠️ Spyder IDE: File-by-File Execution Guide

This guide is for the professor (or any user) to run this project step-by-step within the **Spyder IDE**.

## 1. Environment Setup
Before running files in Spyder, ensure all dependencies are installed in your Python environment:

1. Open your terminal or Anaconda Prompt.
2. Navigate to the project folder.
3. Run: `pip install -r requirements.txt`

> [!IMPORTANT]
> Make sure Spyder is using the same Python interpreter where you installed these requirements. You can check this in Spyder under **Tools → Preferences → Python interpreter**.

## 2. Set the Working Directory ⬅️ CRITICAL
This project uses relative paths. For the scripts to find the data, Spyder's "Working Directory" **must** be set to the project root folder.

- **Option A**: Click the folder icon in the top right of Spyder and select the project folder.
- **Option B**: Right-click the project folder in Spyder's "File Explorer" pane and select "Set as working directory".

## 3. Recommended Running Order
You can run these scripts individually by opening them in Spyder and clicking the **Run** button (or pressing `F5`).

| Step | File | Purpose |
| :--- | :--- | :--- |
| **0** | `phase0_recon.py` | Validates the dataset structure and checks for missing labels. |
| **1** | `phase1_clean.py` | Cleans labels and converts non-JPEG images to `.jpg`. |
| **2** | `phase2_eda.py` | Generates exploratory plots in the `eda_outputs/` folder. |
| **3** | `phase3_aug.py` | Performs data augmentation for minority classes. |
| **4** | `train_yolo.py` | Trains the YOLOv8m model. (Requires GPU for speed). |
| **5** | `phase5_analysis.py`| Generates comparative performance charts. |
| **App** | `app.py` | **Note**: This file should be run via terminal: `streamlit run app.py` |

## 4. Handling "Module Not Found" Errors
If you see "Module Not Found", it means the library is not installed in the environment Spyder is currently using. 

1. Check your installation and install missing ones using `pip install <module_name>`.
2. Consistently use the same environment for both installation and execution.

## 5. Summary of Files
- **Preprocessing**: `restructure_data.py`, `preprocess_data.py`.
- **Training**: `train_yolo.py`, `train_detr.py`.
- **Inference**: `inference.py` (runs a quick test on a few images).
- **Visualization**: `visualize_labels.py` (draws boxes on sample images to check alignment).
