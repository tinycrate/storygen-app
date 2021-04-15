
Please refer to the instructions below to get started locally.

=== INSTRUCTIONS ===

To run the application locally:

  1. Extract the fine-tuned models to the root of this folder 
	- The model might be in different submission from the source
  2. Make sure Python version 3.8 or above is installed
	- Older version might work but not guaranteed
  3. Run run_local.bat
  4. Wait for the packages to download
  5. A webpage will popup, wait for the server to startup
	- If the webpage does not pop up, visit http://127.0.0.1:5000
	- Make sure the port 5000 is not occupied by other applications

===================

Additional information:

The application can be hosted as a web service by running server.py. The
The web server already hosts a production version of the client application
and connecting it the server's root will bring up the client's webpage.

The ideal way of hosting this application is to hide it behind a reverse proxy
by using web hosting server such as NGINX. You might also want to serve the 
client files inside the "static" folder seperately since the Flask API are not 
optimized to serve large files to a large amount of clients.

