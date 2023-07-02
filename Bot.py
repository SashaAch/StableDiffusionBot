import os
import webuiapi
from PIL import Image
import io
import asyncio
from telebot.async_telebot import AsyncTeleBot
import numpy as np
from collections import deque
from telebot import types
import pillow_heif
import dotenv
import traceback

dotenv.load_dotenv()  # take environment variables from .env.
bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
host = os.getenv("TELEGRAM_BOT_HOST")

bot = AsyncTeleBot(bot_token)
api = webuiapi.WebUIApi(host=host, port=8080, steps=20)    # Create API client
upscale_factor = 1  # Default upscaling factor

# Define a queue for upscaling tasks
queue = deque()


user_states = {}
user_styles = {}
controlnet_mod = {}
task_data = {}

# Получаем список стилей
styles_list = ['Ghibli', 'Vector Illustrations', 'Digital/Oil Painting', 'Indie Game', 'Original Photo Style', 'Black and White Film Noir', 'Isometric Rooms', 'Space Hologram', 'Cute Creatures', 'Realistic Photo Portraits', 'Professional Scenic Photographs', 'Manga', 'Anime', 'illustration', 'Caricature', 'Comic book', 'Cinematic', 'Cinematic (horror)', 'Cinematic (art)', 'Gloomy', 'Painting', 'Midjourney (warm)', 'Midjourney', 'XpucT']

controlnet_modules = {'depth':'depth_leres++', 'openpose':'openpose_full', 'canny':'canny'}
controlnet_models = {'depth':'control_v11f1p_sd15_depth [cfd03158]',
                     'openpose':'control_v11p_sd15_openpose [cab727d4]',
                     'canny':'control_v11p_sd15_canny [d14c016b]'}

# Define a dictionary to keep track of each user's position in the queue
user_positions = {}
def create_main_markup():
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    itembtn1 = types.KeyboardButton('txt2img')
    itembtn2 = types.KeyboardButton('img2img')
    itembtn3 = types.KeyboardButton('Upscale')
    markup.add(itembtn1, itembtn2, itembtn3)
    return markup

def create_txt2img_markup():
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    select_style_btn = types.KeyboardButton('Select style')
    clear_styles_btn = types.KeyboardButton('Clear styles')
    repeat_generation_btn = types.KeyboardButton('Repeat with same prompt')
    back_btn = types.KeyboardButton('Back in main menu')
    markup.add(select_style_btn, clear_styles_btn, repeat_generation_btn, back_btn)
    return markup


def create_styles_markup(chat_id):
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    # Add the styles as buttons
    if user_states[chat_id] == 'style_transfer':
        back_btn = types.KeyboardButton('Back in style transfer menu')
    elif user_states[chat_id] == 'txt2img':
        back_btn = types.KeyboardButton('Back in txt2img menu')

    markup.add(back_btn)
    for style in styles_list:
        style_btn = types.KeyboardButton(style)
        markup.add(style_btn)

    return markup

def create_img2img_markup():
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    style_transfer_btn = types.KeyboardButton('Style Transfer')
    # upscale_img2img_btn = types.KeyboardButton('Upscale img2img')
    back_btn = types.KeyboardButton('Back in main menu')
    # markup.add(style_transfer_btn, upscale_img2img_btn, back_btn)
    markup.add(style_transfer_btn, back_btn)
    return markup

def create_style_transfer_markup():
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    select_style_btn = types.KeyboardButton('Select style')
    clear_styles_btn = types.KeyboardButton('Clear styles')
    repeat_generation_btn = types.KeyboardButton('Repeat with same prompt')
    openpose_btn = types.KeyboardButton('OpenPose')
    depth_btn = types.KeyboardButton('Depth')
    canny_btn = types.KeyboardButton('Canny')
    back_btn = types.KeyboardButton('Back in img2img menu')
    markup.add(select_style_btn,clear_styles_btn, repeat_generation_btn, openpose_btn, depth_btn, canny_btn, back_btn)
    return markup

def create_upscale_img2img_markup():
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    upscale_factor_btn = types.KeyboardButton('Select upscale factor')
    back_btn = types.KeyboardButton('Back in img2img menu')
    markup.add(upscale_factor_btn, back_btn)
    return markup

def create_upscale_markup():
    markup = types.ReplyKeyboardMarkup(resize_keyboard=True)
    itembtn1 = types.KeyboardButton('1.5x')
    itembtn2 = types.KeyboardButton('2x')
    itembtn3 = types.KeyboardButton('2.5x')
    itembtn4 = types.KeyboardButton('3x')
    itembtn5 = types.KeyboardButton('3.5x')
    itembtn6 = types.KeyboardButton('4x')
    back_btn = types.KeyboardButton('Back in main menu')
    markup.row(itembtn1, itembtn2, itembtn3)
    markup.row(itembtn4, itembtn5, itembtn6)
    markup.add(back_btn)
    return markup

@bot.message_handler(commands=['start'])
async def start(message):
    markup = create_main_markup()
    await bot.send_message(message.chat.id, "Welcome! Please select an option:", reply_markup=markup)
    user_states[message.chat.id] = 'None'

async def handle_back_in_main_menu(chat_id):
    markup = create_main_markup()
    await bot.send_message(chat_id, "Back to main menu. Please select an option:", reply_markup=markup)
    user_states[chat_id] = None

async def handle_back_in_txt2img_menu(chat_id):
    markup = create_txt2img_markup()
    await bot.send_message(chat_id, "Back to txt2img menu. Please select an option:", reply_markup=markup)

async def handle_back_in_img2img_menu(chat_id):
    markup = create_img2img_markup()
    await bot.send_message(chat_id, "Back to img2img menu. Please select an option:", reply_markup=markup)
    user_states[chat_id] = 'img2img'

async def handle_back_in_style_transfer_menu(chat_id):
    markup = create_style_transfer_markup()
    await bot.send_message(chat_id, "Back to style transfer menu. Please select an option:", reply_markup=markup)
    user_states[chat_id] = 'style_transfer'

async def handle_select_style(chat_id):
    markup = create_styles_markup(chat_id)
    await bot.send_message(chat_id, "Select a style:", reply_markup=markup)


async def handle_clear_styles(chat_id):
    user_styles[chat_id] = ['Default_Negative (sfw)','negative_promt_XpucT' ]
    await bot.send_message(chat_id, "Styles cleared. You can now select new styles.")

message_handlers = {
        'Back in main menu': handle_back_in_main_menu,
        'Back in txt2img menu': handle_back_in_txt2img_menu,
        'Back in img2img menu': handle_back_in_img2img_menu,
        'Back in style transfer menu': handle_back_in_style_transfer_menu,
        'Select style': handle_select_style,
        'Clear styles': handle_clear_styles
    }

@bot.message_handler(content_types=['text'])
async def handle_text(message):
    global upscale_factor
    chat_id = message.chat.id

    if chat_id not in task_data:
        task_data[chat_id] = {'prompt': None}
    # Check the user state
    user_state = user_states.get(chat_id)

    if user_state == 'txt2img':
        await handle_txt2img_state(message, chat_id)

    elif user_state == 'img2img':
        await handle_img2img_state(message, chat_id)

    elif user_state == 'style_transfer':
        await handle_style_transfer_state(message, chat_id)

    elif user_state == 'upscale_img2img':
        await handle_upscale_img2img_state(message, chat_id)

    else:
        await handle_default_state(message, chat_id)


async def handle_task(message, chat_id, task_type):
    last_task_data = task_data.get(chat_id, {'prompt': ''})

    if chat_id not in user_styles:
        user_styles[chat_id] = ['negative_promt_XpucT', 'Default_Negative (sfw)']

    if message.text in message_handlers:
        await message_handlers[message.text](chat_id)
    elif message.text == 'Repeat with same prompt':
        await handle_repeat_prompt(message, chat_id, task_type, last_task_data)
    elif message.text in styles_list:
        await handle_style_choice(message, chat_id)
    else:
        await handle_new_prompt(message, chat_id, task_type)


async def handle_repeat_prompt(message, chat_id, task_type, last_task_data):
    if last_task_data['prompt']:
        prompt_text = last_task_data['prompt']

        selected_style = user_styles.get(chat_id)
        new_task_data = {
            'message': message,
            'prompt': prompt_text,
            'style': selected_style,
            'type': task_type,
        }

        try:
            if user_states[chat_id] == 'style_transfer':
                if 'img' not in new_task_data:
                    new_task_data['module'] = 'depth'
                    if 'img' not in task_data[chat_id]:
                        raise ValueError("Image not provided.")
                elif 'module' in task_data[chat_id]:
                    new_task_data['module'] = task_data[chat_id]['module']


            task_data[chat_id].update(new_task_data)
            queue.append(task_data[chat_id])
            user_positions[chat_id] = len(queue)

            await bot.reply_to(
                message,
                f"Repeating with the same prompt: {prompt_text}. Your position in the queue is {user_positions[chat_id]}. Image generation will start soon, please wait..."
            )
        except ValueError:
            await bot.send_message(
                chat_id,
                "You haven't sent a photo. Please upload a photo for style transfer."
            )
    else:
        await bot.reply_to(
            message,
            "No previous prompt found. Please type the text you want to generate to image."
        )


async def handle_style_choice(message, chat_id):
    user_styles[chat_id].append(message.text)
    selected_style = user_styles.get(chat_id)
    styles_str = ', '.join(i for i in selected_style[2:])

    await bot.send_message(
        chat_id, f"Your styles: {styles_str}. Style selected. Now type the text you want to generate to image."
        )


async def handle_new_prompt(message, chat_id, task_type):
    prompt_text = message.text

    selected_style = user_styles.get(chat_id)
    styles_str = ', '.join(i for i in selected_style[2:])

    new_task_data = {
        'message': message,
        'prompt': prompt_text,
        'style': selected_style,
        'type': task_type,
    }

    try:
        if user_states[chat_id] == 'style_transfer':
            if 'img' not in new_task_data:
                new_task_data['module'] = 'depth'
                if 'img' not in task_data[chat_id]:
                    raise ValueError("Image not provided.")
            elif 'module' in task_data[chat_id]:
                new_task_data['module'] = task_data[chat_id]['module']

        task_data[chat_id].update(new_task_data)

        queue.append(task_data[chat_id])
        user_positions[message.chat.id] = len(queue)

        await bot.reply_to(
            message,
            f"Your styles: {styles_str}. Your position in the queue is {user_positions[message.chat.id]}. Image generation will start soon, please wait..."
        )
    except ValueError:
        await bot.send_message(
            chat_id,
            "You haven't sent a photo. Please upload a photo for style transfer."
        )


async def handle_txt2img_state(message, chat_id):
    await handle_task(message, chat_id, 'txt2img')


async def handle_style_transfer_state(message, chat_id):
    if message.text in ['OpenPose', 'Depth', 'Canny']:
        module = {'module': message.text.lower()}

        task_data[chat_id].update(module)
        await bot.send_message(
            chat_id,
            f"You choose: Style Transfer from {message.text}. Please type the text or load image to style transfer"
        )
    else:
        await handle_task(message, chat_id, 'img2img_style')


async def handle_img2img_state(message, chat_id):
    if message.text == 'Style Transfer':
        markup = create_style_transfer_markup()

        instructions = (
            "You are in the Style Transfer menu. Here's what you can do:\n"
            "1. Select Style: Choose a style that you want to apply to your image.\n"
            "2. Select Method: Choose a method for style transfer. By default, 'Depth' is selected. Other options are 'OpenPose' and 'Canny'.\n"
            "3. Upload Photo: Send a photo (in compressed or document format) to which you want to apply the selected style.\n"
            "4. Repeat with the same prompt: If you want, you can repeat the style transfer with the same settings and a different image.\n"
            "5. Clear styles: Clear the selected styles if you want to start over.\n"
            "6. Go Back: You can go back to the previous menu by clicking 'Back in img2img menu'.\n"
            "\n"
            "Please start by selecting a style or uploading a photo."
        )

        await bot.send_message(chat_id, instructions, reply_markup=markup)
        user_states[chat_id] = 'style_transfer'


    elif message.text == 'Upscale img2img':
        markup = create_upscale_img2img_markup()
        await bot.send_message(chat_id, "You are in Upscale img2img menu.", reply_markup=markup)
        user_states[chat_id] = 'upscale_img2img'

    elif message.text == 'Back in main menu':
        markup = create_main_markup()
        await bot.send_message(chat_id, "Back in main menu.", reply_markup=markup)
        user_states[chat_id] = None


async def handle_upscale_img2img_state(message, chat_id):
    if message.text == 'Select upscale factor':
        # Обработка выбора коэффициента увеличения
        pass

    elif message.text == 'Back in img2img menu':
        markup = create_img2img_markup()
        await bot.send_message(chat_id, "Back in img2img menu.", reply_markup=markup)
        user_states[chat_id] = 'img2img'

async def handle_default_state(message, chat_id):
    global upscale_factor
    if message.text == 'txt2img':
        markup = create_txt2img_markup()
        await bot.send_message(chat_id, "You are in txt2img menu. Please type the text you want to convert to image or select style.", reply_markup=markup)
        user_states[chat_id] = 'txt2img'
    elif message.text == 'img2img':
        markup = create_img2img_markup()
        await bot.send_message(chat_id, "You are in img2img menu. Please select an option:", reply_markup=markup)
        user_states[chat_id] = 'img2img'

    elif message.text == 'Upscale':
        markup = create_upscale_markup()
        await bot.send_message(chat_id, "You are in Upscale menu. Please select an upscale factor and send a photo (in compressed or document format):", reply_markup=markup)
        user_states[chat_id] = 'Upscale'

    elif message.text in ['1.5x', '2x', '2.5x', '3x', '3.5x', '4x']:
        upscale_factor = float(message.text.rstrip('x'))
        await bot.send_message(
            chat_id, f"Set upscaling factor to {upscale_factor}. Now, send the image you want to upscale."
        )
    elif message.text in message_handlers:
        await message_handlers[message.text](chat_id)



@bot.callback_query_handler(func=lambda call: True)
async def callback_query(call):
    if call.data == 'set_upscale_factor':
        markup = create_upscale_markup()
        await bot.send_message(call.message.chat.id, "Please select a new upscale factor:", reply_markup=markup)


@bot.message_handler(content_types=['photo'])
async def enqueue_image(message):
    global upscale_factor
    global task_data

    chat_id = message.chat.id

    # Если пользователь в режиме 'style_transfer' (img2img)
    if chat_id in user_states and user_states[chat_id] == 'style_transfer':
        # Получить информацию о файле и загрузить изображение
        file_info = await bot.get_file(message.photo[-1].file_id)
        downloaded_file = await bot.download_file(file_info.file_path)
        image = Image.open(io.BytesIO(downloaded_file))

        if image.width > 750 or image.height > 750:
            image = image.resize((int(image.width * 0.6), int(image.height * 0.6)))

        task_image = {'img': image}

        task_data[chat_id].update(task_image)


        await bot.reply_to(message, f"Image uploaded. Continue selecting options for style transfer.")

    # Если пользователь в режиме 'Upscale'
    elif chat_id in user_states and user_states[chat_id] == 'Upscale':
        # Получить информацию о файле и загрузить изображение
        file_info = await bot.get_file(message.photo[-1].file_id)
        downloaded_file = await bot.download_file(file_info.file_path)
        image = Image.open(io.BytesIO(downloaded_file))

        # Добавить задачу в очередь и обновить позицию пользователя
        task_data = {
            'message': message,
            'image': image,
            'type': 'upscale',  # указать тип задачи
        }
        queue.append(task_data)
        user_positions[chat_id] = len(queue)

        # Отправить сообщение пользователю
        await bot.reply_to(
            message,
            f"Your position in the queue is {user_positions[chat_id]}. Upscaling will start soon, please wait..."
            )

    else:
        # В случае, если пользователь не находится в каком-либо конкретном режиме
        pass


@bot.message_handler(content_types=['document'])
async def enqueue_document(message):
    global upscale_factor
    global task_data
    chat_id = message.chat.id
    file_info = await bot.get_file(message.document.file_id)

    try:
        downloaded_file = await bot.download_file(file_info.file_path)
        file_extension = file_info.file_path.split('.')[-1].lower()  # Get the file extension

        if file_extension == 'heic':
            heif_file = pillow_heif.open_heif(io.BytesIO(downloaded_file))
            image_data = np.asarray(heif_file[0])  # access first frame of image
            image = Image.fromarray(image_data)
        else:
            image = Image.open(io.BytesIO(downloaded_file))

        width, height = image.size

        if width > 2560 or height > 1440:
            # Calculate the aspect ratio
            aspect_ratio = width / height

            if aspect_ratio > 1:
                # Landscape image
                new_width = min(width, 2560)
                new_height = int(new_width / aspect_ratio)
            else:
                # Portrait image
                new_height = min(height, 1440)
                new_width = int(new_height * aspect_ratio)

            # Resize the image
            image = image.resize((new_width, new_height), Image.LANCZOS)

    except IOError as e:
        await bot.reply_to(message, "The document is not a recognized image. Please send an image file.")
        return

    # Добавляем проверку на user_state
    if chat_id in user_states and user_states[chat_id] == 'style_transfer':
        # Здесь можно обработать загрузку документа как изображения в режиме style_transfer
        task_image = {'img': image}
        task_data[chat_id].update(task_image)
        await bot.reply_to(message, f"Document image uploaded. Continue selecting options for style transfer.")

    elif chat_id in user_states and user_states[chat_id] == 'Upscale':
        # Add the task to the queue and update the user's position
        task_data = {
            'message': message,
            'image': image,
            'type': 'upscale',  # specify the type of task
        }
        queue.append(task_data)
        user_positions[message.chat.id] = len(queue)

        await bot.reply_to(
            message,
            f"Your position in the queue is {user_positions[message.chat.id]}. Upscaling will start soon, please wait..."
        )

    else:
        # В случае, если пользователь не находится в каком-либо конкретном режиме
        pass


async def process_queue():
    while True:
        if queue:
            task_queue_data = queue.popleft()

            message = task_queue_data['message']
            try:

                chat_id = message.chat.id
                for chat_id in user_positions.keys():
                    user_positions[chat_id] -= 1

                if task_queue_data['type'] == 'upscale':
                    await bot.reply_to(message, f"Upscaling started with {upscale_factor}x scale, please wait...")

                    image = task_queue_data['image']
                    result = await api.extra_single_image(
                        image=image,
                        upscaler_1='R-ESRGAN General WDN 4xV3',
                        upscaling_resize=upscale_factor,
                        use_async=True
                    )

                    # Send the upscaled image back to the user
                    output_image_stream = io.BytesIO()
                    result.image.save(output_image_stream, format='PNG')
                    output_image_stream.seek(0)

                    markup = types.InlineKeyboardMarkup()
                    markup.add(types.InlineKeyboardButton('Set Upscale Factor', callback_data='set_upscale_factor'))
                    await bot.send_document(
                        chat_id, output_image_stream, caption='Here is your upscaled image.', reply_markup=markup,
                        visible_file_name='upscaled_image.png'
                    )

                elif task_queue_data['type'] == 'txt2img':
                    await bot.reply_to(message, f"Generating image, please wait...")

                    prompt = task_queue_data['prompt']
                    style = task_queue_data['style']

                    result = await api.txt2img(
                        prompt=prompt,
                        styles=style,
                        cfg_scale=7,
                        width=512,
                        height=512,
                        use_async=True
                    )

                    # Send the generated image back to the user
                    output_image_stream = io.BytesIO()
                    result.image.save(output_image_stream, format='PNG')
                    output_image_stream.seek(0)

                    await bot.send_document(
                        chat_id, output_image_stream, caption='Here is your generated image.',
                        visible_file_name='generated_image.png'
                    )

                elif task_queue_data['type'] == 'img2img_style':
                    await bot.reply_to(message, f"Style Transfering, please wait...")


                    prompt = task_queue_data['prompt']
                    style = task_queue_data['style']
                    module = task_queue_data['module']
                    img = task_queue_data['img']

                    unit2 = None

                    if module == 'canny':
                        unit2 = webuiapi.ControlNetUnit(
                            input_image=img, module=controlnet_modules[module], model=controlnet_models[module],
                            resize_mode="Resize and Fill", threshold_a=60, pixel_perfect=True,
                            threshold_b=170
                        )
                        unit1 = webuiapi.ControlNetUnit(
                            input_image=img, module='depth', model='depth',
                            resize_mode="Resize and Fill", threshold_a=0, pixel_perfect=True,
                            threshold_b=0
                        )

                    else:
                        unit1 = webuiapi.ControlNetUnit(
                            input_image=img, module=controlnet_modules[module], model=controlnet_models[module],
                            resize_mode="Resize and Fill",threshold_a = 0, pixel_perfect=True,
                             threshold_b= 0 )

                    result = await api.img2img(
                        prompt=prompt,
                        images=[img],
                        width=img.width,
                        height=img.height,
                        controlnet_units=[unit1] if unit2 is None else [unit1, unit2],
                        sampler_name="Euler a",
                        cfg_scale=7,
                        styles=style,
                        resize_mode=3,
                        use_async=True
                        )


                    # Send the generated image back to the user
                    output_image_stream = io.BytesIO()
                    result.image.save(output_image_stream, format='PNG')
                    output_image_stream.seek(0)

                    await bot.send_document(
                        chat_id, output_image_stream, caption='Here is your generated image.',
                        visible_file_name='generated_image.png'
                    )
            except Exception as ex:
                print(traceback.format_exc())
                markup = create_main_markup()
                await bot.reply_to(message, f"An error occurred, please try again.",
                                   reply_markup=markup)


        await asyncio.sleep(1)


async def run_bot():
    retry_delay = 5  # 5 seconds
    while True:
        try:
            await bot.polling()

        except:

            print(f"Error while setting up bot polling: {traceback.format_exc()}")
            # Retry after a delay
            await asyncio.sleep(retry_delay)


async def main():
    await asyncio.gather(run_bot(), process_queue())

# Запуск асинхронной функции main
asyncio.run(main())