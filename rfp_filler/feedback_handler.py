#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RFP Feedback Handler
- Detects user modifications in the 'Answer' or 'Comments' columns of a pre-filled RFP Excel file
- Logs or exports the feedback for further fine-tuning or analysis
"""

import openpyxl
import os
import pandas as pd


def detect_feedback(original_path, user_path, output_path=None):
    """
    Compares the original pre-filled Excel and the user-modified Excel.
    Returns a DataFrame of rows where 'Answer' or 'Comments' was changed by the user.
    """
    orig_df = pd.read_excel(original_path)
    user_df = pd.read_excel(user_path)
    feedback_rows = []
    for idx, (orig_row, user_row) in enumerate(zip(orig_df.itertuples(index=False), user_df.itertuples(index=False)), start=2):
        orig_answer = getattr(orig_row, 'Answer', '')
        user_answer = getattr(user_row, 'Answer', '')
        orig_comment = getattr(orig_row, 'Comments', '')
        user_comment = getattr(user_row, 'Comments', '')
        if orig_answer != user_answer or orig_comment != user_comment:
            feedback_rows.append({
                'Row': idx,
                'Question': getattr(user_row, 'Questions', ''),
                'Original_Answer': orig_answer,
                'User_Answer': user_answer,
                'Original_Comments': orig_comment,
                'User_Comments': user_comment
            })
    feedback_df = pd.DataFrame(feedback_rows)
    if output_path:
        feedback_df.to_excel(output_path, index=False)
    return feedback_df


def main():
    import sys
    if len(sys.argv) < 3:
        print("Usage: python feedback_handler.py <original_prefilled.xlsx> <user_modified.xlsx> [output_feedback.xlsx]")
        return
    original_path = sys.argv[1]
    user_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else None
    feedback_df = detect_feedback(original_path, user_path, output_path)
    print(f"Found {len(feedback_df)} feedback rows.")
    if output_path:
        print(f"Feedback exported to: {output_path}")
    else:
        print(feedback_df)


if __name__ == "__main__":
    main()
