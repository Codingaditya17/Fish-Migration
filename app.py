import streamlit as st
from PIL import Image
import numpy as np
import joblib
import folium
from streamlit_folium import folium_static
import geopandas as gpd
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pickle
from statsmodels.tsa.arima.model import ARIMAResults


# Model loading functions
@st.cache_resource
def load_classification_model():
    model = joblib.load('cnn.pkl')
    return model


@st.cache_resource
def load_location_models(species_name):
    """Load ARIMA models for a specific species"""
    try:
        # base_dir = 'FishMigration/fish_models'
        base_dir = 'C:\\Users\\91902\\Downloads\\FishMigration\\fish_models'

        species_filename = species_name.replace(' ', '_')

        # Load the models and metadata
        lat_model = ARIMAResults.load(f'{base_dir}\\{species_filename}_20241118_lat.pkl')
        lon_model = ARIMAResults.load(f'{base_dir}\\{species_filename}_20241118_lon.pkl')

        with open(f'{base_dir}/{species_filename}_20241118_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)

        return {
            'lat_model': lat_model,
            'lon_model': lon_model,
            'metadata': metadata
        }
    except Exception as e:
        st.error(f"Error loading location models: {str(e)}")
        return None


# Image preprocessing function
def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((128, 128))
    image = np.array(image)
    if image.shape[-1] != 3:
        image = np.stack((image,) * 3, axis=-1)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image


# # Prediction functions
# def predict_class(image: Image.Image, model):
#     preprocessed_image = preprocess_image(image)
#     prediction = model.predict(preprocessed_image)
#     if prediction.size > 0:
#         return np.argmax(prediction, axis=1)[0]
#     return None

def predict_class(image, model):
    # Resize image to match model input size
    image = image.resize((224, 224))  # Fix here: from (128, 128) to (224, 224)

    # Convert to array and normalize
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)  # Shape becomes (1, 224, 224, 3)

    # Predict
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction)

    return predicted_class



def predict_locations(species_name, num_past_days=30, num_future_days=30):
    """Predict past and future locations for a species"""
    models = load_location_models(species_name)
    if not models:
        return None

    lat_model = models['lat_model']
    lon_model = models['lon_model']
    metadata = models['metadata']
    last_known_date = metadata['historical_stats']['last_date']

    # Generate dates
    past_dates = pd.date_range(end=last_known_date, periods=num_past_days)
    future_dates = pd.date_range(start=last_known_date + timedelta(days=1), periods=num_future_days)

    # Generate predictions
    future_lats = lat_model.forecast(steps=num_future_days)
    future_lons = lon_model.forecast(steps=num_future_days)

    # Create DataFrames
    past_data = pd.DataFrame({
        'date': past_dates,
        'latitude': metadata['historical_stats']['lat_mean'],
        'longitude': metadata['historical_stats']['lon_mean'],
        'type': 'historical'
    })

    future_data = pd.DataFrame({
        'date': future_dates,
        'latitude': future_lats,
        'longitude': future_lons,
        'type': 'predicted'
    })

    return pd.concat([past_data, future_data])


def create_map(location_data):
    """Create an interactive map with historical and predicted paths"""
    # Calculate center of the map
    center_lat = location_data['latitude'].mean()
    center_lon = location_data['longitude'].mean()

    # Create base map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=6)

    # Plot historical path
    historical = location_data[location_data['type'] == 'historical']
    future = location_data[location_data['type'] == 'predicted']

    # Add historical path
    points = list(zip(historical['latitude'], historical['longitude']))
    folium.PolyLine(
        points,
        weight=2,
        color='blue',
        opacity=0.8,
        popup='Historical Path'
    ).add_to(m)

    # Add predicted path
    points = list(zip(future['latitude'], future['longitude']))
    folium.PolyLine(
        points,
        weight=2,
        color='red',
        opacity=0.8,
        popup='Predicted Path'
    ).add_to(m)

    # Add markers for start and end points
    folium.Marker(
        [historical.iloc[0]['latitude'], historical.iloc[0]['longitude']],
        popup='Start',
        icon=folium.Icon(color='green')
    ).add_to(m)

    folium.Marker(
        [future.iloc[-1]['latitude'], future.iloc[-1]['longitude']],
        popup='Predicted End',
        icon=folium.Icon(color='red')
    ).add_to(m)

    return m


# Class labels dictionary
class_labels = {
    0: "Black Sea Sprat",
    1: "Gilt-Head Bream",
    2: "Hourse Mackerel",
    3: "Red Mullet",
    4: "Red Sea Bream",
    5: "Sea Bass",
    6: "Shrimp",
    7: "Striped Red Mullet",
    8: "Trout"
}

# Species to scientific name mapping
species_scientific_names = {
    "Black Sea Sprat": "Sprattus sprattus",
    "Gilt-Head Bream": "Sparus aurata",
    "Hourse Mackerel": "Oncorhynchus keta",
    "Red Mullet": "Mullus surmuletus",
    "Red Sea Bream": "Pagrus pagrus",
    "Sea Bass": "Dicentrarchus labrax",
    "Striped Red Mullet": "Trachurus trachurus",
    "Trout": "Oncorhynchus mykiss"
}

fish_species_features = {
    "Trout": 
    """\n
    - Forked tail and streamlined, medium-width body.\n
    - Thrive in fast-flowing freshwater environments.\n
    - Coloration ranges from olive green to silvery.\n
    - Black spots scattered along the back and fins.\n
    - Some species (e.g., rainbow trout) have a pink lateral stripe.\n
    - Small, smooth scales with relatively soft texture.\n
    """,

    "Striped Red Mullet": 
    """\n
    - Slightly forked tail and narrow, elongated body.\n
    - Reddish-pink coloration with prominent yellow longitudinal stripes.\n
    - Long barbels under the chin used to search for food in the seabed.\n
    """,

    "Shrimp": 
    """\n
    - Fan-like tail and slender, curved body.\n
    - Hard exoskeleton; body is segmented.\n
    - Color ranges from translucent pink to brown depending on species/habitat.\n
    - Long antennae and multiple walking legs.\n
    - Agile swimmers; vital to aquatic food chains.\n
    """,

    "Sea Bass": 
    """\n
    - Slightly forked tail with a robust, broad body.\n
    - Silver-gray coloration with a darker back.\n
    - May have a dark spot near the gills.\n
    - Firm flesh and spiny dorsal fins.\n
    - Adaptable to both saltwater and freshwater environments.\n
    """,

    "Red Sea Bream": 
    """\n
    - Forked tail and medium-width, high-backed body.\n
    - Distinctive reddish-pink color with shimmering scales.\n
    - Slightly pointed snout and steep forehead.\n
    - Valued commercially and in cuisine for its delicate flavor.\n
    """,

    "Red Mullet": 
    """\n
    - Slightly forked tail and slim, elongated body.\n
    - Reddish with a golden hue; may have faint stripes.\n
    - Chin barbels used for sensing prey in seabeds.\n
    - Flesh is flavorful and considered a delicacy.\n
    """,

    "Horse Mackerel": 
    """\n
    - Deeply forked tail and narrow, torpedo-shaped body.\n
    - Metallic blue-green back with silver flanks.\n
    - Distinct lateral line with small bony plates (scutes).\n
    - Fast swimmers; commonly found in schools.\n
    """,

    "Gilt-Head Bream": 
    """\n
    - Forked tail and broad, oval-shaped body.\n
    - Silver body with golden band between the eyes.\n
    - Golden spot on each cheek.\n
    - Prized in Mediterranean cuisine for rich, white flesh.\n
    """,

    "Black Sea Sprat": 
    """\n
    - Forked tail and very slim, small body.\n
    - Silvery coloration with a darker back.\n
    - Smooth, reflective scales.\n
    - Found in large schools; prey for larger fish and seabirds.\n
    - Crucial role in marine food web.\n
    """
}



# Streamlit app
def main():
    st.title("Fish Species Classification and Location Prediction")
    st.write("Upload an image of a fish to classify it and predict its migration path")

    # Load classification model
    classification_model = load_classification_model()

    # Image upload
    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg"])

    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Classify image
        st.write("Classifying...")
        predicted_class = predict_class(image, classification_model)

        if predicted_class is not None:
            # Get predicted species
            species_common_name = class_labels.get(predicted_class, 'Unknown')
            st.success(f"Predicted Species: {species_common_name}")

            # Get scientific name
            scientific_name = species_scientific_names.get(species_common_name)
            features_string = fish_species_features.get(species_common_name)
            if scientific_name:
                st.write(f"Scientific Name: {scientific_name}")
                st.write(f"Features: {features_string}")
                # Add date selection for prediction
                st.write("### Location Prediction Settings")
                num_past_days = st.slider("Number of past days to show", 7, 90, 30)
                num_future_days = st.slider("Number of future days to predict", 7, 90, 30)

                if st.button("Show Migration Path"):
                    with st.spinner("Generating migration path..."):
                        # Get location predictions
                        location_data = predict_locations(
                            scientific_name,
                            num_past_days=num_past_days,
                            num_future_days=num_future_days
                        )

                        if location_data is not None:
                            # Create tabs for different visualizations
                            tab1, tab2 = st.tabs(["Interactive Map", "Movement Analysis"])

                            with tab1:
                                st.write("### Fish Migration Path")
                                # Create and display map
                                m = create_map(location_data)
                                folium_static(m)

                            with tab2:
                                st.write("### Movement Analysis")

                                # Create movement analysis plots
                                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

                                # Plot latitude changes
                                location_data.plot(
                                    x='date',
                                    y='latitude',
                                    ax=ax1,
                                    color=['blue' if t == 'historical' else 'red' for t in location_data['type']]
                                )
                                ax1.set_title('Latitude Over Time')
                                ax1.set_ylabel('Latitude')

                                # Plot longitude changes
                                location_data.plot(
                                    x='date',
                                    y='longitude',
                                    ax=ax2,
                                    color=['blue' if t == 'historical' else 'red' for t in location_data['type']]
                                )
                                ax2.set_title('Longitude Over Time')
                                ax2.set_ylabel('Longitude')

                                plt.tight_layout()
                                st.pyplot(fig)

                                # Display data table
                                st.write("### Detailed Location Data")
                                st.dataframe(location_data)
                        else:
                            st.error(
                                "Could not generate location predictions. Please check if models are available for this species.")
            else:
                st.warning("Location prediction is not available for this species.")
        else:
            st.error("Could not classify the image.")


if __name__ == "__main__":
    main()
