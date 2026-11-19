window.RAMVI_PACK = {
  "character": {
    "name": "RAMVI Character Pack",
    "version": "1.0",
    "canvas": [
      768,
      768
    ],
    "format": "RGBA PNG",
    "original_source": "source/ramvi_original_logo.png",
    "character_base": "parts/body/visor_without_features.png",
    "eye_parts": [
      "parts/eyes/normal.png",
      "parts/eyes/half.png",
      "parts/eyes/closed.png",
      "parts/eyes/happy.png",
      "parts/eyes/surprised.png",
      "parts/eyes/wink.png",
      "parts/eyes/angry.png",
      "parts/eyes/look_left.png",
      "parts/eyes/look_right.png",
      "parts/eyes/sleep.png"
    ],
    "mouth_parts": [
      "parts/mouths/neutral.png",
      "parts/mouths/smile.png",
      "parts/mouths/open.png",
      "parts/mouths/wide.png",
      "parts/mouths/o.png",
      "parts/mouths/grin.png",
      "parts/mouths/frown.png",
      "parts/mouths/tiny.png",
      "parts/mouths/talk_a.png",
      "parts/mouths/talk_e.png"
    ],
    "expressions": {
      "neutral": {
        "eyes": "normal",
        "mouth": "neutral",
        "file": "sprites/expressions/neutral.png"
      },
      "confident": {
        "eyes": "normal",
        "mouth": "smile",
        "file": "sprites/expressions/confident.png"
      },
      "cheerful": {
        "eyes": "happy",
        "mouth": "wide",
        "file": "sprites/expressions/cheerful.png"
      },
      "laugh": {
        "eyes": "happy",
        "mouth": "open",
        "file": "sprites/expressions/laugh.png"
      },
      "surprise": {
        "eyes": "surprised",
        "mouth": "o",
        "file": "sprites/expressions/surprise.png"
      },
      "wink": {
        "eyes": "wink",
        "mouth": "smile",
        "file": "sprites/expressions/wink.png"
      },
      "angry": {
        "eyes": "angry",
        "mouth": "frown",
        "file": "sprites/expressions/angry.png"
      },
      "sad": {
        "eyes": "half",
        "mouth": "frown",
        "file": "sprites/expressions/sad.png"
      },
      "sleepy": {
        "eyes": "sleep",
        "mouth": "neutral",
        "file": "sprites/expressions/sleepy.png"
      },
      "thinking": {
        "eyes": "look_left",
        "mouth": "tiny",
        "file": "sprites/expressions/thinking.png"
      },
      "talking": {
        "eyes": "normal",
        "mouth": "talk_a",
        "file": "sprites/expressions/talking.png"
      },
      "excited": {
        "eyes": "surprised",
        "mouth": "wide",
        "file": "sprites/expressions/excited.png"
      },
      "side_eye": {
        "eyes": "look_right",
        "mouth": "smile",
        "file": "sprites/expressions/side_eye.png"
      }
    },
    "turns": {
      "front": "sprites/turns/front.png",
      "three_quarter_left": "sprites/turns/three_quarter_left.png",
      "profile_left": "sprites/turns/profile_left.png",
      "three_quarter_right": "sprites/turns/three_quarter_right.png",
      "profile_right": "sprites/turns/profile_right.png"
    },
    "animations": {
      "idle": {
        "frames": [
          "animations/idle/000.png",
          "animations/idle/001.png",
          "animations/idle/002.png",
          "animations/idle/003.png",
          "animations/idle/004.png",
          "animations/idle/005.png",
          "animations/idle/006.png",
          "animations/idle/007.png",
          "animations/idle/008.png",
          "animations/idle/009.png",
          "animations/idle/010.png",
          "animations/idle/011.png"
        ],
        "duration_ms": 115,
        "loop": true,
        "canvas": [
          768,
          768
        ]
      },
      "blink": {
        "frames": [
          "animations/blink/000.png",
          "animations/blink/001.png",
          "animations/blink/002.png",
          "animations/blink/003.png",
          "animations/blink/004.png",
          "animations/blink/005.png"
        ],
        "duration_ms": 100,
        "loop": true,
        "canvas": [
          768,
          768
        ]
      },
      "talk": {
        "frames": [
          "animations/talk/000.png",
          "animations/talk/001.png",
          "animations/talk/002.png",
          "animations/talk/003.png",
          "animations/talk/004.png",
          "animations/talk/005.png",
          "animations/talk/006.png",
          "animations/talk/007.png"
        ],
        "duration_ms": 95,
        "loop": true,
        "canvas": [
          768,
          768
        ]
      },
      "nod": {
        "frames": [
          "animations/nod/000.png",
          "animations/nod/001.png",
          "animations/nod/002.png",
          "animations/nod/003.png",
          "animations/nod/004.png",
          "animations/nod/005.png",
          "animations/nod/006.png",
          "animations/nod/007.png",
          "animations/nod/008.png"
        ],
        "duration_ms": 115,
        "loop": true,
        "canvas": [
          768,
          768
        ]
      },
      "shake": {
        "frames": [
          "animations/shake/000.png",
          "animations/shake/001.png",
          "animations/shake/002.png",
          "animations/shake/003.png",
          "animations/shake/004.png",
          "animations/shake/005.png",
          "animations/shake/006.png",
          "animations/shake/007.png",
          "animations/shake/008.png",
          "animations/shake/009.png"
        ],
        "duration_ms": 115,
        "loop": true,
        "canvas": [
          768,
          768
        ]
      }
    },
    "notes": [
      "Front-facing expressions and animations are layered edits of the uploaded RAMVI mascot.",
      "Turns are simulated 2D transforms, not redrawn side/back character art.",
      "Concept board is a design reference image, not layered PSD/SVG."
    ]
  },
  "sounds": {
    "pack": "RAMVI Character Sound Pack",
    "version": "1.0.0",
    "audio_type": "original procedural synthesized sound effects",
    "sample_rate_hz": 48000,
    "channels": 1,
    "bit_depth": 16,
    "count": 69,
    "license": "CC0-1.0; original sounds generated for this pack",
    "files": [
      {
        "id": "character/apology",
        "file": "character/apology.wav",
        "category": "character",
        "description_ru": "Извинение: виноватый нисходящий мотив",
        "duration_s": 0.59,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "react"
        ]
      },
      {
        "id": "character/beep_answer",
        "file": "character/beep_answer.wav",
        "category": "character",
        "description_ru": "Утвердительный роботизированный ответ",
        "duration_s": 0.51,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "voice",
          "answer"
        ]
      },
      {
        "id": "character/beep_no",
        "file": "character/beep_no.wav",
        "category": "character",
        "description_ru": "Отрицание: низкие сигналы",
        "duration_s": 0.43,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "voice",
          "no"
        ]
      },
      {
        "id": "character/beep_question",
        "file": "character/beep_question.wav",
        "category": "character",
        "description_ru": "Вопросительный роботизированный сигнал",
        "duration_s": 0.43,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "voice",
          "question"
        ]
      },
      {
        "id": "character/chuckle_01",
        "file": "character/chuckle_01.wav",
        "category": "character",
        "description_ru": "Мультяшный электронный смешок",
        "duration_s": 0.52,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "laugh"
        ]
      },
      {
        "id": "character/chuckle_02",
        "file": "character/chuckle_02.wav",
        "category": "character",
        "description_ru": "Медленный тихий электронный смешок",
        "duration_s": 0.843,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "laugh"
        ]
      },
      {
        "id": "character/giggle",
        "file": "character/giggle.wav",
        "category": "character",
        "description_ru": "Весёлое короткое хихиканье",
        "duration_s": 0.42,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "laugh"
        ]
      },
      {
        "id": "character/hiccup",
        "file": "character/hiccup.wav",
        "category": "character",
        "description_ru": "Мультяшная электронная икота",
        "duration_s": 0.33,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "react"
        ]
      },
      {
        "id": "character/sigh",
        "file": "character/sigh.wav",
        "category": "character",
        "description_ru": "Вздох мультяшного робота",
        "duration_s": 0.72,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "sigh"
        ]
      },
      {
        "id": "character/sleep_mumble",
        "file": "character/sleep_mumble.wav",
        "category": "character",
        "description_ru": "Сонное электронное бормотание без слов",
        "duration_s": 0.95,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "sleep",
          "mumble"
        ]
      },
      {
        "id": "character/talk_long_01",
        "file": "character/talk_long_01.wav",
        "category": "character",
        "description_ru": "Длинная псевдоречевая реплика",
        "duration_s": 1.858,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "talk",
          "long"
        ]
      },
      {
        "id": "character/talk_medium_01",
        "file": "character/talk_medium_01.wav",
        "category": "character",
        "description_ru": "Средняя псевдоречевая реплика",
        "duration_s": 0.854,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "talk",
          "medium"
        ]
      },
      {
        "id": "character/talk_medium_02",
        "file": "character/talk_medium_02.wav",
        "category": "character",
        "description_ru": "Средняя низкая псевдоречевая реплика",
        "duration_s": 1.1,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "talk",
          "medium"
        ]
      },
      {
        "id": "character/talk_short_01",
        "file": "character/talk_short_01.wav",
        "category": "character",
        "description_ru": "Короткая псевдоречевая реплика",
        "duration_s": 0.24,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "talk",
          "short"
        ]
      },
      {
        "id": "character/talk_short_02",
        "file": "character/talk_short_02.wav",
        "category": "character",
        "description_ru": "Короткая низкая псевдоречевая реплика",
        "duration_s": 0.3,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "talk",
          "short"
        ]
      },
      {
        "id": "character/thinking_hum",
        "file": "character/thinking_hum.wav",
        "category": "character",
        "description_ru": "Задумчивое роботизированное мычание",
        "duration_s": 0.68,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "thinking",
          "hum"
        ]
      },
      {
        "id": "character/wake_up",
        "file": "character/wake_up.wav",
        "category": "character",
        "description_ru": "Просыпание персонажа",
        "duration_s": 0.63,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "wake"
        ]
      },
      {
        "id": "emotions/angry",
        "file": "emotions/angry.wav",
        "category": "emotions",
        "description_ru": "Злость: резкое электронное ворчание",
        "duration_s": 0.42,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "angry"
        ]
      },
      {
        "id": "emotions/calm",
        "file": "emotions/calm.wav",
        "category": "emotions",
        "description_ru": "Спокойствие: мягкий двойной тон",
        "duration_s": 0.83,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "calm"
        ]
      },
      {
        "id": "emotions/confused",
        "file": "emotions/confused.wav",
        "category": "emotions",
        "description_ru": "Непонимание: неуверенный мотив",
        "duration_s": 0.58,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "confused"
        ]
      },
      {
        "id": "emotions/curious",
        "file": "emotions/curious.wav",
        "category": "emotions",
        "description_ru": "Любопытство: короткий восходящий вопрос",
        "duration_s": 0.45,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "curious"
        ]
      },
      {
        "id": "emotions/displeased",
        "file": "emotions/displeased.wav",
        "category": "emotions",
        "description_ru": "Недовольство: два нисходящих тона",
        "duration_s": 0.57,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "negative"
        ]
      },
      {
        "id": "emotions/embarrassed",
        "file": "emotions/embarrassed.wav",
        "category": "emotions",
        "description_ru": "Смущение: два неловких сигнала",
        "duration_s": 0.38,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "embarrassed"
        ]
      },
      {
        "id": "emotions/excited",
        "file": "emotions/excited.wav",
        "category": "emotions",
        "description_ru": "Восторг: быстрая растущая трель",
        "duration_s": 0.63,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "excited"
        ]
      },
      {
        "id": "emotions/happy_01",
        "file": "emotions/happy_01.wav",
        "category": "emotions",
        "description_ru": "Радость: три восходящих мультяшных тона",
        "duration_s": 0.69,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "happy",
          "positive"
        ]
      },
      {
        "id": "emotions/happy_02",
        "file": "emotions/happy_02.wav",
        "category": "emotions",
        "description_ru": "Радость: двойной звонкий возглас",
        "duration_s": 0.36,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "happy"
        ]
      },
      {
        "id": "emotions/proud",
        "file": "emotions/proud.wav",
        "category": "emotions",
        "description_ru": "Гордость: маленькая победная мелодия",
        "duration_s": 0.86,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "proud"
        ]
      },
      {
        "id": "emotions/sad",
        "file": "emotions/sad.wav",
        "category": "emotions",
        "description_ru": "Грусть: плавный нисходящий мотив",
        "duration_s": 0.62,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "sad"
        ]
      },
      {
        "id": "emotions/sleepy",
        "file": "emotions/sleepy.wav",
        "category": "emotions",
        "description_ru": "Сонливость: медленный мягкий спад",
        "duration_s": 0.88,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "sleepy"
        ]
      },
      {
        "id": "emotions/surprised_01",
        "file": "emotions/surprised_01.wav",
        "category": "emotions",
        "description_ru": "Удивление: резкий скачок высоты",
        "duration_s": 0.63,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "surprised"
        ]
      },
      {
        "id": "emotions/surprised_02",
        "file": "emotions/surprised_02.wav",
        "category": "emotions",
        "description_ru": "Удивление: протяжное «о?» без речи",
        "duration_s": 0.4,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "surprised"
        ]
      },
      {
        "id": "emotions/thinking_01",
        "file": "emotions/thinking_01.wav",
        "category": "emotions",
        "description_ru": "Размышление: три вопросительных звука",
        "duration_s": 0.7,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "thinking"
        ]
      },
      {
        "id": "emotions/thinking_02",
        "file": "emotions/thinking_02.wav",
        "category": "emotions",
        "description_ru": "Размышление: тихий задумчивый сигнал",
        "duration_s": 0.79,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "thinking"
        ]
      },
      {
        "id": "motion/appear_01",
        "file": "motion/appear_01.wav",
        "category": "motion",
        "description_ru": "Появление персонажа",
        "duration_s": 0.6,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "appear"
        ]
      },
      {
        "id": "motion/appear_02",
        "file": "motion/appear_02.wav",
        "category": "motion",
        "description_ru": "Волшебное появление персонажа",
        "duration_s": 0.56,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "appear"
        ]
      },
      {
        "id": "motion/blink_01",
        "file": "motion/blink_01.wav",
        "category": "motion",
        "description_ru": "Мягкое моргание",
        "duration_s": 0.08,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "blink"
        ]
      },
      {
        "id": "motion/blink_02",
        "file": "motion/blink_02.wav",
        "category": "motion",
        "description_ru": "Тихое моргание",
        "duration_s": 0.055,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "blink"
        ]
      },
      {
        "id": "motion/bob",
        "file": "motion/bob.wav",
        "category": "motion",
        "description_ru": "Лёгкое подпрыгивание в idle",
        "duration_s": 0.33,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "idle",
          "movement"
        ]
      },
      {
        "id": "motion/disappear",
        "file": "motion/disappear.wav",
        "category": "motion",
        "description_ru": "Исчезновение персонажа",
        "duration_s": 0.55,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "disappear"
        ]
      },
      {
        "id": "motion/double_blink",
        "file": "motion/double_blink.wav",
        "category": "motion",
        "description_ru": "Двойное моргание",
        "duration_s": 0.285,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "blink"
        ]
      },
      {
        "id": "motion/idle_chime",
        "file": "motion/idle_chime.wav",
        "category": "motion",
        "description_ru": "Редкий тихий idle-сигнал",
        "duration_s": 0.79,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "idle"
        ]
      },
      {
        "id": "motion/land",
        "file": "motion/land.wav",
        "category": "motion",
        "description_ru": "Мягкое приземление",
        "duration_s": 0.16,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "movement",
          "land"
        ]
      },
      {
        "id": "motion/move_down",
        "file": "motion/move_down.wav",
        "category": "motion",
        "description_ru": "Спуск персонажа",
        "duration_s": 0.29,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "movement"
        ]
      },
      {
        "id": "motion/move_left",
        "file": "motion/move_left.wav",
        "category": "motion",
        "description_ru": "Движение влево",
        "duration_s": 0.21,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "movement"
        ]
      },
      {
        "id": "motion/move_right",
        "file": "motion/move_right.wav",
        "category": "motion",
        "description_ru": "Движение вправо",
        "duration_s": 0.21,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "movement"
        ]
      },
      {
        "id": "motion/move_up",
        "file": "motion/move_up.wav",
        "category": "motion",
        "description_ru": "Подъём персонажа",
        "duration_s": 0.27,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "movement"
        ]
      },
      {
        "id": "motion/nod",
        "file": "motion/nod.wav",
        "category": "motion",
        "description_ru": "Кивок",
        "duration_s": 0.32,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "nod"
        ]
      },
      {
        "id": "motion/shake_head",
        "file": "motion/shake_head.wav",
        "category": "motion",
        "description_ru": "Покачивание головой",
        "duration_s": 0.405,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "shake"
        ]
      },
      {
        "id": "motion/turn",
        "file": "motion/turn.wav",
        "category": "motion",
        "description_ru": "Поворот персонажа",
        "duration_s": 0.2,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "movement",
          "turn"
        ]
      },
      {
        "id": "ui/click_crisp",
        "file": "ui/click_crisp.wav",
        "category": "ui",
        "description_ru": "Чёткий щелчок кнопки",
        "duration_s": 0.05,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "click",
          "button"
        ]
      },
      {
        "id": "ui/click_soft",
        "file": "ui/click_soft.wav",
        "category": "ui",
        "description_ru": "Мягкий щелчок кнопки",
        "duration_s": 0.048,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "click",
          "button"
        ]
      },
      {
        "id": "ui/close_panel",
        "file": "ui/close_panel.wav",
        "category": "ui",
        "description_ru": "Закрытие окна или панели",
        "duration_s": 0.24,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "window",
          "close"
        ]
      },
      {
        "id": "ui/confirm_01",
        "file": "ui/confirm_01.wav",
        "category": "ui",
        "description_ru": "Подтверждение: два позитивных тона",
        "duration_s": 0.58,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "success",
          "confirm"
        ]
      },
      {
        "id": "ui/confirm_02",
        "file": "ui/confirm_02.wav",
        "category": "ui",
        "description_ru": "Подтверждение: тройной сигнал",
        "duration_s": 0.53,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "success",
          "confirm"
        ]
      },
      {
        "id": "ui/error_01",
        "file": "ui/error_01.wav",
        "category": "ui",
        "description_ru": "Ошибка: нисходящий двойной сигнал",
        "duration_s": 0.47,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "error"
        ]
      },
      {
        "id": "ui/error_02",
        "file": "ui/error_02.wav",
        "category": "ui",
        "description_ru": "Ошибка: повторяющийся низкий тон",
        "duration_s": 0.56,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "error"
        ]
      },
      {
        "id": "ui/hover",
        "file": "ui/hover.wav",
        "category": "ui",
        "description_ru": "Наведение курсора",
        "duration_s": 0.065,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "hover"
        ]
      },
      {
        "id": "ui/loading_tick",
        "file": "ui/loading_tick.wav",
        "category": "ui",
        "description_ru": "Тихий индикатор обработки",
        "duration_s": 0.09,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "loading"
        ]
      },
      {
        "id": "ui/message_received",
        "file": "ui/message_received.wav",
        "category": "ui",
        "description_ru": "Получено сообщение",
        "duration_s": 0.48,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "message"
        ]
      },
      {
        "id": "ui/notification_soft",
        "file": "ui/notification_soft.wav",
        "category": "ui",
        "description_ru": "Мягкое уведомление",
        "duration_s": 0.67,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "notification"
        ]
      },
      {
        "id": "ui/notification_urgent",
        "file": "ui/notification_urgent.wav",
        "category": "ui",
        "description_ru": "Срочное уведомление",
        "duration_s": 0.5,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "notification"
        ]
      },
      {
        "id": "ui/open_panel",
        "file": "ui/open_panel.wav",
        "category": "ui",
        "description_ru": "Открытие окна или панели",
        "duration_s": 0.26,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "window",
          "open"
        ]
      },
      {
        "id": "ui/send_message",
        "file": "ui/send_message.wav",
        "category": "ui",
        "description_ru": "Отправлено сообщение",
        "duration_s": 0.17,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "message"
        ]
      },
      {
        "id": "ui/start_task",
        "file": "ui/start_task.wav",
        "category": "ui",
        "description_ru": "Запуск задания",
        "duration_s": 0.58,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "task",
          "start"
        ]
      },
      {
        "id": "ui/task_complete",
        "file": "ui/task_complete.wav",
        "category": "ui",
        "description_ru": "Успешное завершение задания",
        "duration_s": 0.63,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "task",
          "success"
        ]
      },
      {
        "id": "ui/task_failed",
        "file": "ui/task_failed.wav",
        "category": "ui",
        "description_ru": "Неудачное завершение задания",
        "duration_s": 0.59,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "task",
          "error"
        ]
      },
      {
        "id": "ui/toggle_off",
        "file": "ui/toggle_off.wav",
        "category": "ui",
        "description_ru": "Переключатель выключен",
        "duration_s": 0.265,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "toggle"
        ]
      },
      {
        "id": "ui/toggle_on",
        "file": "ui/toggle_on.wav",
        "category": "ui",
        "description_ru": "Переключатель включён",
        "duration_s": 0.26,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "toggle"
        ]
      },
      {
        "id": "ui/warning",
        "file": "ui/warning.wav",
        "category": "ui",
        "description_ru": "Предупреждение: два коротких сигнала",
        "duration_s": 0.4,
        "sample_rate": 48000,
        "channels": 1,
        "format": "PCM_16",
        "tags": [
          "warning"
        ]
      }
    ]
  },
  "checked_images": 84
};
