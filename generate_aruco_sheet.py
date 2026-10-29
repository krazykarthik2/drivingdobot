"""
Generates 4 printable ArUco markers (DICT_4X4_50, IDs 0, 1, 2, 3) with labels and border guides.
Run this script to produce 'aruco_markers_printable.png' or individual marker images.
"""
import cv2
import numpy as np
import os

def generate_printable_markers(output_sheet="aruco_markers_printable.png"):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    
    # 200x200 px marker image (with 100px margin around)
    marker_size = 240
    border_bits = 1
    
    card_w, card_h = 360, 420
    sheet = np.full((card_h * 2 + 60, card_w * 2 + 60, 3), 255, dtype=np.uint8)

    positions = [
        (0, "TOP-LEFT (ID: 0)", "Dobot Frame: [X = +300mm, Y = -150mm]", 0, 0),
        (1, "TOP-RIGHT (ID: 1)", "Dobot Frame: [X = +300mm, Y = +150mm]", 0, 1),
        (2, "BOTTOM-RIGHT (ID: 2)", "Dobot Frame: [X = +150mm, Y = +150mm]", 1, 1),
        (3, "BOTTOM-LEFT (ID: 3)", "Dobot Frame: [X = +150mm, Y = -150mm]", 1, 0),
    ]

    for marker_id, label, coord_label, r, c in positions:
        marker_img = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_size, borderBits=border_bits)
        
        card = np.full((card_h, card_w, 3), 255, dtype=np.uint8)
        # Draw cutting outline
        cv2.rectangle(card, (5, 5), (card_w - 5, card_h - 5), (180, 180, 180), 1)
        
        # Center marker in card
        offset_x = (card_w - marker_size) // 2
        offset_y = 50
        card[offset_y:offset_y + marker_size, offset_x:offset_x + marker_size] = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)

        # Labels
        cv2.putText(card, label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(card, coord_label, (15, card_h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1, cv2.LINE_AA)
        cv2.putText(card, "Center is Calibration Anchor (+)", (35, card_h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 120), 1, cv2.LINE_AA)

        # Place on sheet
        top_y = 30 + r * (card_h + 20)
        left_x = 30 + c * (card_w + 20)
        sheet[top_y:top_y + card_h, left_x:left_x + card_w] = card
        
        # Save individual marker file too
        cv2.imwrite(f"aruco_id_{marker_id}.png", card)

    cv2.imwrite(output_sheet, sheet)
    print(f"[OK] Generated printable marker sheet: {os.path.abspath(output_sheet)}")
    print(f"[OK] Generated individual marker files: aruco_id_0.png through aruco_id_3.png")

if __name__ == "__main__":
    generate_printable_markers()
