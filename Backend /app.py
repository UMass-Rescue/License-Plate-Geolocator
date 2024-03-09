from flask import Flask

# Create an instance of Flask
app = Flask(__name__)

# Define a route for the home page
@app.route('/')
def index():
    return 'Hello, World! This is the home page.'

# Define a route for other pages
@app.route('/about')
def about():
    return 'About page'

# Run the application
if __name__ == '__main__':
    app.run(debug=True)
