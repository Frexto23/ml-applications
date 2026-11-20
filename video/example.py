prompt = """
A photorealistic cinematic luxury fashion video at night in a beautiful modern city with neon lights.

A beautiful adult woman with long straight jet-black hair, pale porcelain skin, striking facial features and a confident expression is sitting inside a luxurious black Lamborghini parked on a wet city street.

She is wearing a tight elegant black mini dress, glossy black high-heeled boots and stylish accessories. The car has beautiful glossy black paint, sharp headlights and luxurious leather interior.

The camera starts with a cinematic three-quarter view of the Lamborghini. The driver's door slowly opens. The woman gracefully steps out of the car, places one foot on the wet pavement, stands up and gently closes the door.

She adjusts her long black hair with one hand, looks directly into the camera with a confident expression, then slowly walks towards the camera with elegant natural movements.

Her hair moves gently as she walks. The camera smoothly moves backwards, keeping her full body and the Lamborghini visible in the background.

Beautiful neon reflections on the wet asphalt, cinematic city lights, realistic car reflections, luxury fashion commercial, natural human movement, realistic facial expressions, detailed skin texture, professional cinematography, shallow depth of field, 35mm lens, photorealistic live-action footage, smooth motion, one continuous shot, no cuts."""

negative_prompt = """blurry, low quality, distorted face, deformed body, extra arms, extra legs, extra fingers, bad hands, malformed feet, unnatural walking, floating objects, duplicate woman, duplicate car, distorted car, changing car shape, flickering, jittering, static image, frozen movement, cartoon, anime, oversaturated colors, subtitles, text, watermark, camera shake"""

import sora67

gen = sora67.VideoGenerator()

gen.generate_and_save(prompt, negative_prompt)