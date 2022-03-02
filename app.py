from flask import Flask, request, jsonify
from models import StrukNet
import urllib.request

app = Flask(__name__)
model = StrukNet()

@app.errorhandler(400)
def bad_request(e):
	return jsonify({"status": "not ok", "message": "this server could not understand your request"}), 400

@app.errorhandler(404)
def not_found(e):
	return jsonify({"status": "not found", "message": "route not found"}), 404

@app.errorhandler(500)
def not_found(e):
    return jsonify({"status": "internal error", "message": "internal error"}), 500

@app.route('/detect', methods=['GET', 'POST'])
def detect_object():
	if request.method == 'GET':
		if request.args.get('url'):
			with urllib.request.urlopen(request.args.get('url')) as url:
				return jsonify({"status": "ok", "result": model.infer(url.read())}), 200
		else:
			return jsonify({"status": "bad request", "message": "Url is not present"}), 400
	elif request.method == 'POST':
		if request.files.get('image'):
			return jsonify({"status": "ok", "result": model.infer(request.files['image'].read())}), 200
		else:
			return jsonify({"status": "bad request", "message": "image is not present"}), 400   
		
	
if __name__ == '__main__':
	app.run(host='0.0.0.0')