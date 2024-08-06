import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image
st.set_page_config(
    page_title = 'Emotion Detection - EDA',
    layout = 'wide',
    initial_sidebar_state = 'expanded'
)

def visualize_samples_by_label(df, label, num_samples=20):
    samples = df[df['label'] == label]['images'].iloc[:num_samples].tolist()
    num_cols = min(num_samples, 5)
    num_rows = (num_samples - 1) // num_cols + 1
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(10, 2 * num_rows))
    count = 0
    for i in range(num_rows):
        for j in range(num_cols):
            if count < len(samples):
                sample = samples[count]
                img = cv2.imread(sample)
                ax = axes[i, j]
                # ax.set_title(sample.split('/')[-1].split('\')[-1])
                ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                ax.axis('off')
                count += 1
    plt.tight_layout()
    plt.show()

def run():
    st.title('Identify Your Emotion')
    # Add a picture
    st.image('https://sloanreview.mit.edu/wp-content/uploads/2019/11/FR-Motro-Measuring-Emotion-Facial-Recognition-Employee-Productivity-2400-1290x860.jpg')
    st.write('### Objective')
    st.markdown('---')
    st.write("To create an immersive and personalized gaming experience for customers, a gaming company wants to integrate emotion recognition technology into their games. Initially, they have tasked me, as a data engineer, with developing a deep learning model to identify customer emotions such as anger, disgust, fear, happiness, neutrality, sadness, and surprise from images. To achieve this objective, we will employ SMART analysis to define clear and actionable goals, develop a deep learning model for emotion identification, use accuracy as a metric to evaluate the model's performance, and utilize transfer learning to address limited data and improve the model's effectiveness. By following these steps, we aim to develop a robust emotion recognition system for an enhanced gaming experience.")
    st.markdown('---')
    # Add a description
    st.write("# ***:red[Let's Explore it]***")
    # Make a line
    st.markdown('---')
    st.write('### ***Angry***')
    st.image('angry.png', caption='Angry')
    st.write("This is a 48 x 48 pixel gray scale of angry images that contained the people who display a clenched jaw, intense eye contact, furrowed brows, and reddened skin.")
    st.markdown('---')
    st.write('### ***Disgusted***')
    st.image('disgusted.png', caption='disgusted')
    st.write("This is a 48 x 48 pixel gray scale of disgusted images that contained the people who display the lips curling into a snarling frown, the nostrils flaring, the cheeks pushing up to squinting lower lids and a furrowed brow.")
    st.markdown('---')
    st.write('### ***Fearful***')
    st.image('fearful.png', caption='fearful')
    st.write("This is a 48 x 48 pixel gray scale of fearful images that contained the people who display the wide opened eyes and flared nostrils")
    st.markdown('---')
    st.write('### ***Happy***')
    st.image('happy.png', caption='happy')
    st.write("This is a 48 x 48 pixel gray scale of happy images that contained the people who display a raised eyebrows, dimple cheeks, raised cheeks, and open mouth")
    st.markdown('---')
    st.write('### ***Neutral***')
    st.image('neutral.png', caption='neutral')
    st.write("This is a 48 x 48 pixel gray scale of neutral images that contained the people who express no particular emotion")
    st.markdown('---')
    st.write('### ***Sad***')
    st.image('sad.png', caption='sad')
    st.write("This is a 48 x 48 pixel gray scale of sad images that contained the people who display owering the corners of the mouth, allowing the eyebrows to descend, and drooping the eyelids")
    st.markdown('---')
    st.write('### ***Surprised***')
    st.image('surprised.png', caption='surprised')
    st.write("This is a 48 x 48 pixel gray scale of surprised images that contained the people who display a eyebrow raised, but not drawn together, upper eyelids raised, lower eyelids neutral, and jaw dropped down.")
    st.markdown('---')
    st.write("### ***EDA Analysis***")
    st.markdown('---')
    st.write('From Images that display 7 emotions, i can see some resemblance between happy and surprised, sad and fearful. From that instances, i can predict that the model is having a hard time to differentiate between each class')
    st.write('---')
if __name__ == '__main__':
  run()