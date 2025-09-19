"""
Handles reading and parsing of YAML configuration files.
"""
import yaml
from pathlib import Path
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class YAMLHandler:
    """A simple handler for reading and parsing YAML files."""

    def load_fields(self, config_path: Path) -> List[Dict[str, Any]]:
        """
        Load field configurations from a YAML file.

        Args:
            config_path: The path to the YAML configuration file.

        Returns:
            A list of dictionaries, where each dictionary represents a field.

        Raises:
            FileNotFoundError: If the config file does not exist.
            ValueError: If the YAML content is not in the expected format.
        """
        logger.info(f"Loading field configuration from {config_path}")
        try:
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f)

            if not isinstance(data, list):
                raise ValueError("YAML content should be a list of fields.")

            # Basic validation
            for item in data:
                if not isinstance(item, dict) or "field_name" not in item or "description" not in item:
                    raise ValueError("Each item in the YAML file must be a dictionary with 'field_name' and 'description'.")

            logger.info(f"Successfully loaded {len(data)} fields.")
            return data
        except FileNotFoundError:
            logger.error(f"Fields configuration file not found at: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {config_path}: {e}")
            raise
        except ValueError as e:
            logger.error(f"Invalid format in YAML file {config_path}: {e}")
            raise
