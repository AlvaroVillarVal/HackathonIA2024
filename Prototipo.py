class AI:
    def __init__(self):
        self.difficulty = 1

    def adapt_difficulty(self, score):
        if score > 10:
            self.difficulty += 1
        elif score < 5:
            self.difficulty -= 1

    def play(self):
        # AI logic based on current difficulty
        pass

# Create an instance of the AI
ai = AI()

# Example usage
ai.adapt_difficulty(8)  # Adjust difficulty based on score
ai.play()  # Perform AI actions based on current difficulty