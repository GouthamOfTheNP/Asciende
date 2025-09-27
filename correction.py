import os
from PIL import Image

class ASLCorrection:

    def __init__(self, dataset_path: str):

        self.dataset_path = dataset_path
        self.reference_images = {}
        self.feedback_dict = {
            "A": "Keep your fingers together and curl the thumb across.",
            "B": "Hold your fingers straight and together, thumb across palm.",
            "C": "Curve your hand to form a C shape.",
            "D": "Touch the tip of the middle, ring, and pinky fingers to the thumb, index finger upright.",
            "E": "Curl your fingers down to touch your thumb, forming a closed shape.",
            "F": "Form an 'OK' sign: thumb and index finger touch, other fingers extended.",
            "G": "Hold your index finger straight, thumb out, other fingers curled.",
            "H": "Hold index and middle fingers straight and together, thumb across palm.",
            "I": "Curl all fingers except pinky, which points up.",
            "J": "Draw a 'J' with your pinky in the air.",
            "K": "Thumb between middle and index, other fingers straight.",
            "L": "Make an 'L' shape with thumb and index.",
            "M": "Place thumb under three fingers.",
            "N": "Place thumb under two fingers.",
            "O": "Form an 'O' with all fingers.",
            "P": "Thumb between middle and index, hand upside down.",
            "Q": "Thumb and index hold something, other fingers tucked.",
            "R": "Cross index and middle finger.",
            "S": "Make a fist, thumb in front.",
            "T": "Thumb under index finger.",
            "U": "Index and middle fingers straight together.",
            "V": "Index and middle fingers spread in a V shape.",
            "W": "Index, middle, and ring fingers spread.",
            "X": "Curl index finger, other fingers in a fist.",
            "Y": "Thumb and pinky extended, other fingers folded.",
            "SPACE": "Hold your palm sideways to indicate space."
        }
        self._load_reference_images()

    def _load_reference_images(self):
        """Load one reference image per letter."""
        for letter in os.listdir(self.dataset_path):
            letter_path = os.path.join(self.dataset_path, letter)
            if os.path.isdir(letter_path):
                img_files = [f for f in os.listdir(letter_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if img_files:
                    img_path = os.path.join(letter_path, img_files[0])
                    self.reference_images[letter] = Image.open(img_path)

    def get_correction(self, letter: str):
        """
        Get feedback and reference image for a given letter.
        Returns (feedback_text, reference_image)
        """
        feedback = self.feedback_dict.get(letter.upper(), "No feedback available for this letter.")
        reference_image = self.reference_images.get(letter.upper(), None)
        return feedback, reference_image


# Example
if __name__ == "__main__":
    asl_corrector = ASLCorrection("asl_dataset")
    letter = "A"
    feedback, image = asl_corrector.get_correction(letter)
    print(f"Feedback for '{letter}': {feedback}")
    if image:
        image.show()
