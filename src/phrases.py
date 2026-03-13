import random


OPENINGS = [
    "Hello, I'm ready to help. What would you like to know?",
    "Hi there. I'm ready whenever you are. What would you like to ask?",
    "I'm ready to help. What do you want to know?",
]

LISTENING = [
    "I'm listening.",
    "Go ahead, I'm listening.",
    "Tell me what you need.",
]

CHECKING = [
    "Okay, let me check.",
    "Sure, give me a second.",
    "Alright, let me take a look.",
]

FOLLOW_UP = [
    "What would you like to do next?",
    "Let me know if you want me to check again.",
    "You can ask me again whenever you're ready.",
]

SCENE_EMPTY = [
    "I can't see any supported target objects in front of you right now.",
    "I don't see any supported objects clearly right now.",
]

DIRECTION_NO_TARGET = [
    "I can't find the {target}.",
    "I can't see the {target} right now.",
]

SHOW_HAND = [
    "I found the {target}. Show your hand in the camera so I can guide you.",
    "I can see the {target}. Now show me your hand so I can guide you.",
]

GRASP_YES = [
    "Yes, you are holding the {target}. {follow}",
    "Yes, it looks like you have the {target}. {follow}",
]

GRASP_NO = [
    "Not yet. The {target} is still around {clock}, which means {cardinal}. You should {movement}. It is roughly {approx_cm} centimeters away from your hand.",
    "Not yet. The {target} still seems to be near {clock}. Try to {movement}. The gap is about {approx_cm} centimeters.",
]

DIRECTION_REPLY = [
    "The {target} is at about {clock}, which means {cardinal}. You should {movement}. It is roughly {approx_cm} centimeters away, so it feels {distance}. {follow}",
    "Your {target} is around {clock}. In other words, {movement}. It is about {approx_cm} centimeters away and seems {distance}. {follow}",
    "The {target} looks near {clock}, meaning {cardinal}. Try to {movement}. The distance is roughly {approx_cm} centimeters, so it is {distance}. {follow}",
]

SCENE_ONE = [
    "I can see a {objects}. What would you like to pick up?",
    "I can see one object: a {objects}. What would you like to do next?",
]

SCENE_MANY = [
    "I can see {objects}. What would you like to pick up?",
    "I can currently see {objects}. Which one would you like to pick up?",
    "I found {objects}. Tell me which object you want.",
]

UNKNOWN = [
    "Sorry, I didn't understand that. You can ask what objects are in front of you, where an object is, or whether you got it.",
    "Sorry, I missed that. Try asking what you can see, where an object is, or whether you're holding it.",
]


def pick(options):
    return random.choice(options)
