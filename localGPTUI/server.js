const express = require("express");
const axios = require("axios");
const multer = require("multer");
const path = require("path");
const FormData = require("form-data");

const app = express();
const API_HOST = "http://localhost:5110/api";

// Middleware to parse form data
app.use(express.urlencoded({ extended: true }));
app.use(express.json());

// Set EJS as the view engine
app.set("view engine", "ejs");
app.set("views", path.join(__dirname, "views"));

// Set up multer for file uploads
const upload = multer();

// Route for the home page
app.get("/", (req, res) => {
    res.render("home", {
        show_response_modal: false,
        response_dict: { Prompt: "None", Answer: "None", Sources: [["ewf", "wef"]] },
    });
});

// Handle POST requests to "/" for both prompt submissions and file uploads
app.post("/", upload.array("documents"), async (req, res) => {
    try {
        // Check if user_prompt is submitted
        if (req.body.user_prompt) {
            const user_prompt = req.body.user_prompt;
            console.log(`User Prompt: ${user_prompt}`);

            const main_prompt_url = `${API_HOST}/prompt_route`;
            const response = await axios.post(main_prompt_url, { user_prompt });
            console.log(response.status);

            if (response.status === 200) {
                return res.render("home", {
                    show_response_modal: true,
                    response_dict: response.data,
                });
            }
        }

        // Check if documents were uploaded
        if (req.files.length > 0) {
            if (req.body.action === "reset") {
                const delete_source_url = `${API_HOST}/delete_source`;
                await axios.get(delete_source_url);
            }

            const save_document_url = `${API_HOST}/save_document`;
            const run_ingest_url = `${API_HOST}/run_ingest`;

            for (let file of req.files) {
                console.log(`Uploading file: ${file.originalname}`);

                // Create form data for each file
                const formData = new FormData();
                formData.append("document", file.buffer, file.originalname);

                const response = await axios.post(save_document_url, formData, {
                    headers: {
                        ...formData.getHeaders(),
                    },
                });
                console.log(response.status);
            }

            // Call run_ingest after all files are uploaded
            const ingestResponse = await axios.get(run_ingest_url);
            console.log(ingestResponse.status);
        }

        // Render the form for a GET request
        res.render("home", {
            show_response_modal: false,
            response_dict: { Prompt: "None", Answer: "None", Sources: [["ewf", "wef"]] },
        });
    } catch (error) {
        console.error("Error processing request:", error.message);
        res.status(500).send("An error occurred");
    }
});

// Start the server
const PORT = process.env.PORT || 5111;
const HOST = process.env.HOST || "127.0.0.1";

app.listen(PORT, HOST, () => {
    console.log(`Server is running on http://${HOST}:${PORT}`);
});
