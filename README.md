# Pole Classifier
End-to-end pipeline for pole classification using Google Street View images

## Setup
1. **Clone the repository:**
   ```shell
   git clone https://github.com/tharaka-perera/pole_classifier
   cd <repository-directory>
   ```

2. **Create a virtual environment or conda environment:**
    ```shell
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages:**
    ```shell
    pip install -r requirements.txt
    ```

4. **Set up environment variables:**
Create a `.env` file in the root directory and add your API key:
    ```shell
    API_KEY=your_api_key_here
    ```

5. **Copy ML models to `/models` folder and data files to `/data` folder**

## How to Run
```shell
python __main__.py
```






