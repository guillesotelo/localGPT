/* 
Volvo Press Releases Scraper
-----------------------------
Run only locally with `node news_scrapper.js`
Copy file generated and replace the original in API server for ingestion. 
*/

const fs = require('fs');
const path = require('path');
const { Builder, By, Key } = require('selenium-webdriver');
const chrome = require('selenium-webdriver/chrome');
const cheerio = require('cheerio');

const NEWS_URL = "https://www.volvocars.com/intl/media/press-releases/?page=1&pageSize=100";
const OUTPUT_FILE = "AUX_DOCS/volvo_press_releases.txt";

// Initialize Chrome driver
async function initDriver() {
    const options = new chrome.Options();
    options.addArguments(
        "--headless=new",
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--disable-blink-features=AutomationControlled", // basic stealth
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) " +
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.5993.88 Safari/537.36"
    );

    return new Builder()
        .forBrowser('chrome')
        .setChromeOptions(options)
        .build();
}

// Scroll page to load more content
async function scrollToLoad(driver, pause = 2000, maxScrolls = 10) {
    let lastHeight = await driver.executeScript("return document.body.scrollHeight");
    for (let i = 0; i < maxScrolls; i++) {
        await driver.findElement(By.tagName('body')).sendKeys(Key.END);
        await new Promise(r => setTimeout(r, pause));
        let newHeight = await driver.executeScript("return document.body.scrollHeight");
        if (newHeight === lastHeight) break;
        lastHeight = newHeight;
    }
}

// Parse articles using cheerio
function parseArticles(html) {
    const $ = cheerio.load(html);
    const articles = [];

    $('li').each((_, item) => {
        const titleEl = $(item).find('a').first();
        const summaryEl = $(item).find('p').first();
        const dateEl = $(item).find('time');

        const title = titleEl.text().trim().replace(/\s+/g, ' ');
        const summary = summaryEl.text().trim().replace(/\s+/g, ' ');
        const date = dateEl.text().trim();

        if (title && summary && date) {
            articles.push({ title, summary, date });
        }
    });

    return articles;
}

// Load existing articles from file
function loadExistingArticles() {
    if (!fs.existsSync(OUTPUT_FILE)) return new Set();

    const data = fs.readFileSync(OUTPUT_FILE, 'utf-8').split('\n');
    const existing = new Set();

    for (let i = 0; i < data.length; i += 4) {
        if (i + 1 < data.length) {
            const dateLine = data[i].trim();
            // extract date from "Volvo press release {date}:"
            const dateMatch = dateLine.match(/^Volvo press release (.*):$/);
            const date = dateMatch ? dateMatch[1] : dateLine;
            const title = data[i + 1].trim();
            existing.add(`${title}||${date}`);
        }
    }

    return existing;
}

// Save new articles to file
function saveArticles(articles) {
    // Ensure the directory exists
    const dir = path.dirname(OUTPUT_FILE);
    if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
    }

    // Ensure the file exists
    if (!fs.existsSync(OUTPUT_FILE)) {
        fs.writeFileSync(OUTPUT_FILE, '', 'utf-8');
    }

    const existing = loadExistingArticles();
    const newArticles = articles.filter(a => !existing.has(`${a.title}||${a.date}`));

    if (!newArticles.length) {
        console.log("No new articles found.");
        return;
    }

    // Prepend "Volvo press release {date}:" to each paragraph
    const lines = newArticles.flatMap(a => [
        `Volvo press release ${a.date}:`,
        a.title,
        a.summary,
        ''
    ]);

    fs.appendFileSync(OUTPUT_FILE, lines.join('\n') + '\n', 'utf-8');
    console.log(`Saved ${newArticles.length} new articles.`);
}

// Main scraper
async function runScraper() {
    const driver = await initDriver();
    try {
        console.log(`Running driver in ${NEWS_URL}...`)
        await driver.get(NEWS_URL);
        console.log('Scrolling page...')
        await scrollToLoad(driver, 2000, 15);

        console.log('Getting source HTML...')
        const html = await driver.getPageSource();
        const articles = parseArticles(html);
        console.log(`Found ${articles.length} articles.`);

        console.log(`Saving articles in ${OUTPUT_FILE}...`)
        saveArticles(articles);
    } finally {
        await driver.quit();
    }
}

// Run the scraper
runScraper().catch(console.error);
