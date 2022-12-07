from flask import Flask, jsonify, request
import io
import json
import function as usr_src

usr_src.init()

app = Flask(__name__)

# Reference your preloaded global model variable here.
@app.route('/generate', methods=['POST'])
def generate():
    print("request launch")
    if request.method == 'POST':

        # Parse out your arguments
        print("request received")
        prompt = request.form.get('prompt')
        
        if prompt == None:
            return {'message': "No prompt provided"}
        
        print("request processing")
        result = usr_src.inference(prompt.strip())

        # Return the results as a dictionary
        return result



if __name__ == '__main__':
    app.run()