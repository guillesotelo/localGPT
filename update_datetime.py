import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def update_auxiliary_data_file():
    """
    Updates the first two lines of /chatbot/source/api/AUX_DOCS/auxiliary_data.txt
    with the current human-readable and ISO dates.
    Equivalent to the original cron job.
    """
    try:

        file_path = Path("/chatbot/source/api/AUX_DOCS/auxiliary_data.txt")

        # Read the current file content (or create if missing)
        if file_path.exists():
            with file_path.open("r", encoding="utf-8") as f:
                lines = f.readlines()
        else:
            lines = []

        # Ensure file has at least 2 lines
        while len(lines) < 2:
            lines.append("\n")

        # Format dates
        now = datetime.datetime.now()
        human_date = now.strftime("CURRENT_DATE: Today is %A, %B %d, %Y.")
        iso_date = now.strftime("ISO_DATE: %Y-%m-%d")

        # Replace first two lines
        lines[0] = human_date + "\n"
        lines[1] = iso_date + "\n"

        # Write back to file
        with file_path.open("w", encoding="utf-8") as f:
            f.writelines(lines)

        logger.info(f"Updated {file_path} with current dates.")

    except Exception as e:
        # Log the error but don't raise it — this keeps the API running
        logger.error(f"Failed to update date on auxilari_data.txt: {e}", exc_info=True)