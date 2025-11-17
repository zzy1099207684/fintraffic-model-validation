import openai
import glob
import base64
from mimetypes import guess_type
import time

from anyio import sleep

# Model	        Input	Cached input	Output
# gpt-5	        $1.25	$0.125	        $10.00
# gpt-5-mini	$0.25	$0.025	        $2.00
# gpt-4.1	    $2.00	$0.50	        $8.00
# gpt-4.1-mini	$0.40	$0.10	        $1.60

# MODEL_NAME = "gpt-5" temperature can not be zero
# MODEL_NAME = "gpt-5-mini" temperature can not be zero
MODEL_NAME = "gpt-4.1"  # 0.8000
# MODEL_NAME = "gpt-4.1-mini"

openai.api_key = ""


# Function to create a file with the Files API
def create_file(image_path):
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'
    with open(image_path, "rb") as image_file:
        base64_data = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:{mime_type};base64,{base64_data}"


all_count = 0

dry_count = 0
dry_error_count = 0
dry_reject_count = 0

wet_count = 0
wet_error_count = 0
wet_reject_count = 0

snow_cleared_count = 0
snow_cleared_error_count = 0
snow_cleared_reject_count = 0

snow_covered_count = 0
snow_covered_error_count = 0
snow_covered_reject_count = 0

for i in range(4):

    paths = []
    if i == 0:
        paths = glob.glob("picture/dry/*.jpg")
    elif i == 1:
        paths = glob.glob("picture/wet/*.jpg")
    elif i == 2:
        paths = glob.glob("picture/snow_cleared/*.jpeg")
    elif i == 3:
        paths = glob.glob("picture/snow_covered/*.jpeg")
    # Getting the file ID
    for path in paths:
        time.sleep(2)  # Each request is spaced 2 seconds apart to avoid rate limiting
        all_count += 1
        img_id = create_file(path)
        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Output only 0, 1, 2, or 3. Determine the road surface condition in the image: "
                                "0 = dry (no water traces), "
                                "1 = wet (water present but no snow), "
                                "2 = snow cleared (melted snow traces or residual snow marks), "
                                "3 = snow covered. "
                                "If the condition is unclear, select the closest category."
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": img_id, "detail": "auto"}
                        },
                    ],
                }
            ],
        )
        result = response["choices"][0]["message"]["content"]
        if i == 0:  # dry
            if result == '0':
                dry_count += 1
            elif result in ['1', '2', '3']:
                dry_error_count += 1
            else:
                dry_reject_count += 1
        elif i == 1:  # wet
            if result == '1':
                wet_count += 1
            elif result in ['0', '2', '3']:
                wet_error_count += 1
            else:
                wet_reject_count += 1
        elif i == 2:  # snow_cleared
            if result == '2':
                snow_cleared_count += 1
            elif result in ['0', '1', '3']:
                snow_cleared_error_count += 1
            else:
                snow_cleared_reject_count += 1
        elif i == 3:  # snow_covered
            if result == '3':
                snow_covered_count += 1
            elif result in ['0', '1', '2']:
                snow_covered_error_count += 1
            else:
                snow_covered_reject_count += 1
        print(path, "---", result)

# ===== Evaluation Metrics Calculation =====
total_samples = (
        dry_count + dry_error_count + dry_reject_count +
        wet_count + wet_error_count + wet_reject_count +
        snow_cleared_count + snow_cleared_error_count + snow_cleared_reject_count +
        snow_covered_count + snow_covered_error_count + snow_covered_reject_count
)

if total_samples == 0:
    print("No samples, cannot compute metrics.")
else:
    # 1. accuracy
    correct = dry_count + wet_count + snow_cleared_count + snow_covered_count
    accuracy = correct / total_samples


    # Dry (label=0)
    TP_dry = dry_count
    FN_dry = dry_error_count
    FP_dry = wet_error_count + snow_cleared_error_count + snow_covered_error_count

    precision_dry = TP_dry / (TP_dry + dry_error_count) if (TP_dry + dry_error_count) > 0 else 0.0
    recall_dry = TP_dry / (TP_dry + FN_dry) if (TP_dry + FN_dry) > 0 else 0.0
    f1_dry = (2 * precision_dry * recall_dry / (precision_dry + recall_dry)
              if (precision_dry + recall_dry) > 0 else 0.0)

    # Wet (label=1)
    TP_wet = wet_count
    FN_wet = wet_error_count
    precision_wet = TP_wet / (TP_wet + wet_error_count) if (TP_wet + wet_error_count) > 0 else 0.0
    recall_wet = TP_wet / (TP_wet + FN_wet) if (TP_wet + FN_wet) > 0 else 0.0
    f1_wet = (2 * precision_wet * recall_wet / (precision_wet + recall_wet)
              if (precision_wet + recall_wet) > 0 else 0.0)

    # Snow Cleared (label=2)
    TP_snow_cleared = snow_cleared_count
    FN_snow_cleared = snow_cleared_error_count
    precision_snow_cleared = TP_snow_cleared / (TP_snow_cleared + snow_cleared_error_count) if (
                                                                                                       TP_snow_cleared + snow_cleared_error_count) > 0 else 0.0
    recall_snow_cleared = TP_snow_cleared / (TP_snow_cleared + FN_snow_cleared) if (
                                                                                           TP_snow_cleared + FN_snow_cleared) > 0 else 0.0
    f1_snow_cleared = (2 * precision_snow_cleared * recall_snow_cleared / (precision_snow_cleared + recall_snow_cleared)
                       if (precision_snow_cleared + recall_snow_cleared) > 0 else 0.0)

    # Snow Covered (label=3)
    TP_snow_covered = snow_covered_count
    FN_snow_covered = snow_covered_error_count
    precision_snow_covered = TP_snow_covered / (TP_snow_covered + snow_covered_error_count) if (
                                                                                                       TP_snow_covered + snow_covered_error_count) > 0 else 0.0
    recall_snow_covered = TP_snow_covered / (TP_snow_covered + FN_snow_covered) if (
                                                                                           TP_snow_covered + FN_snow_covered) > 0 else 0.0
    f1_snow_covered = (2 * precision_snow_covered * recall_snow_covered / (precision_snow_covered + recall_snow_covered)
                       if (precision_snow_covered + recall_snow_covered) > 0 else 0.0)

    # calculate reject count and rate
    macro_precision = (precision_dry + precision_wet + precision_snow_cleared + precision_snow_covered) / 4
    macro_recall = (recall_dry + recall_wet + recall_snow_cleared + recall_snow_covered) / 4
    macro_f1 = (f1_dry + f1_wet + f1_snow_cleared + f1_snow_covered) / 4

    report_lines = []
    report_lines.append(f"===== Evaluation Metrics By {MODEL_NAME} (4-Class) =====")
    report_lines.append(f"Total samples: {total_samples}")
    report_lines.append(f"Correct predictions: {correct}")
    report_lines.append(f"Accuracy: {accuracy:.4f}")

    report_lines.append("\n-- Dry (label=0) --")
    report_lines.append(f"TP: {TP_dry}, FN: {FN_dry}")
    report_lines.append(f"Precision: {precision_dry:.4f}")
    report_lines.append(f"Recall: {recall_dry:.4f}")
    report_lines.append(f"F1: {f1_dry:.4f}")

    report_lines.append("\n-- Wet (label=1) --")
    report_lines.append(f"TP: {TP_wet}, FN: {FN_wet}")
    report_lines.append(f"Precision: {precision_wet:.4f}")
    report_lines.append(f"Recall: {recall_wet:.4f}")
    report_lines.append(f"F1: {f1_wet:.4f}")

    report_lines.append("\n-- Snow Cleared (label=2) --")
    report_lines.append(f"TP: {TP_snow_cleared}, FN: {FN_snow_cleared}")
    report_lines.append(f"Precision: {precision_snow_cleared:.4f}")
    report_lines.append(f"Recall: {recall_snow_cleared:.4f}")
    report_lines.append(f"F1: {f1_snow_cleared:.4f}")

    report_lines.append("\n-- Snow Covered (label=3) --")
    report_lines.append(f"TP: {TP_snow_covered}, FN: {FN_snow_covered}")
    report_lines.append(f"Precision: {precision_snow_covered:.4f}")
    report_lines.append(f"Recall: {recall_snow_covered:.4f}")
    report_lines.append(f"F1: {f1_snow_covered:.4f}")

    report_lines.append("\n-- Macro Average --")
    report_lines.append(f"Macro Precision: {macro_precision:.4f}")
    report_lines.append(f"Macro Recall: {macro_recall:.4f}")
    report_lines.append(f"Macro F1: {macro_f1:.4f}")


    for line in report_lines:
        print(line)

    with open("picture/GPT_evaluation_report_4class.txt", "a", encoding="utf-8") as f:
        for line in report_lines:
            f.write(line + "\n")
