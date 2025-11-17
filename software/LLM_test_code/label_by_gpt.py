import openai
import glob
import base64
from mimetypes import guess_type
# Model	        Input	Cached input	Output
# gpt-5	        $1.25	$0.125	        $10.00
# gpt-5-mini	$0.25	$0.025	        $2.00
# gpt-4.1	    $2.00	$0.50	        $8.00
# gpt-4.1-mini	$0.40	$0.10	        $1.60

# MODEL_NAME = "gpt-5" temperature can not be zero
# MODEL_NAME = "gpt-5-mini" temperature can not be zero
MODEL_NAME = "gpt-4.1"    #0.8000
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

for i in range(2):
    paths = []
    if i == 0:
        paths = glob.glob("picture/dry/*.jpg")
    elif i == 1:
        paths = glob.glob("picture/wet/*.jpg")
    # Getting the file ID
    for path in paths:
        all_count += 1
        img_id = create_file(path)
        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Output only 0 (dry) and 1 (wet). "
                                "Determine whether the road surface in the picture is wet(1) or dry(0)."
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": img_id, "detail": "auto"}
                        },
                    ],
                }
            ],
            temperature=0
        )
        result = response["choices"][0]["message"]["content"]
        if i == 0:  # dry
            if result == '0':
                dry_count += 1
            elif result == '1':
                dry_error_count += 1
            else:
                dry_reject_count += 1
        elif i == 1:  # wet
            if result == '1':
                wet_count += 1
            elif result == '0':
                wet_error_count += 1
            else:
                wet_reject_count += 1
        print(path,"---",result)

# ===== Evaluation Metrics Calculation =====
total_samples = (
        dry_count + dry_error_count + dry_reject_count +
        wet_count + wet_error_count + wet_reject_count
)

if total_samples == 0:
    print("No samples, cannot compute metrics.")
else:
    # 1. accuracy
    correct = dry_count + wet_count
    accuracy = correct / total_samples

    # 3. Wet as positive class (label=1)
    TP_wet = wet_count
    FP_wet = dry_error_count
    FN_wet = wet_error_count

    precision_wet = TP_wet / (TP_wet + FP_wet) if (TP_wet + FP_wet) > 0 else 0.0
    recall_wet = TP_wet / (TP_wet + FN_wet) if (TP_wet + FN_wet) > 0 else 0.0
    f1_wet = (2 * precision_wet * recall_wet / (precision_wet + recall_wet)
              if (precision_wet + recall_wet) > 0 else 0.0)


    TP_dry = dry_count
    FP_dry = wet_error_count
    FN_dry = dry_error_count

    precision_dry = TP_dry / (TP_dry + FP_dry) if (TP_dry + FP_dry) > 0 else 0.0
    recall_dry = TP_dry / (TP_dry + FN_dry) if (TP_dry + FN_dry) > 0 else 0.0
    f1_dry = (2 * precision_dry * recall_dry / (precision_dry + recall_dry)
              if (precision_dry + recall_dry) > 0 else 0.0)

    report_lines = []
    report_lines.append(f"===== Evaluation Metrics By {MODEL_NAME}=====")
    report_lines.append(f"Total samples: {total_samples}")
    report_lines.append(f"Correct predictions: {correct}")
    report_lines.append(f"Accuracy: {accuracy:.4f}")

    report_lines.append("\n-- Wet (as positive class, label=1) --")
    report_lines.append(f"TP_wet: {TP_wet}, FP_wet: {FP_wet}, FN_wet: {FN_wet}")
    report_lines.append(f"Precision_wet: {precision_wet:.4f}")
    report_lines.append(f"Recall_wet: {recall_wet:.4f}")
    report_lines.append(f"F1_wet: {f1_wet:.4f}")

    report_lines.append("\n-- Dry (as positive class, label=0) --")
    report_lines.append(f"TP_dry: {TP_dry}, FP_dry: {FP_dry}, FN_dry: {FN_dry}")
    report_lines.append(f"Precision_dry: {precision_dry:.4f}")
    report_lines.append(f"Recall_dry: {recall_dry:.4f}")
    report_lines.append(f"F1_dry: {f1_dry:.4f}")


    for line in report_lines:
        print(line)

    with open("log/GPT_evaluation_report.txt", "a", encoding="utf-8") as f:
        for line in report_lines:
            f.write(line + "\n")

