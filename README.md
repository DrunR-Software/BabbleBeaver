# BabbleBeaver

Welcome to BabbleBeaver, an open-source conversational AI platform designed with privacy and flexibility in mind. BabbleBeaver leverages the power of Large Language Models (LLM) to provide customizable and isolated conversational agents, encapsulated in Docker containers for easy deployment and management.

## Overview

BabbleBeaver aims to democratize conversational AI, offering a plug-and-play solution that respects user privacy and data sovereignty. Built on top of FastAPI, BabbleBeaver facilitates rapid development and deployment of AI-powered chatbots with support for multiple, isolated LLM implementations.

## Installation

```bash
git clone https://github.com/YourUsername/BabbleBeaver.git
cd BabbleBeaver
docker-compose up --build
```

Ensure Docker and Docker Compose are installed on your system before running these commands.

## Usage

After installation, BabbleBeaver can be accessed at `http://localhost:8000` by default. The API documentation, generated by FastAPI, is available at `http://localhost:8000/docs`.

## Architecture

BabbleBeaver adopts a modular architecture, with each conversational AI model implemented as a separate submodule. This allows for the easy addition, removal, or replacement of LLM implementations without affecting the core system.

- **Core System**: Built on FastAPI, the core handles API requests, routing, and integration with LLM submodules.
- **LLM Submodules**: Each LLM implementation is encapsulated within its Docker container, ensuring data isolation and security. Submodules communicate with the core system via RESTful APIs or message queues.
- **Data Layer**: For storing conversational logs, user data, and other relevant information, with support for encryption and GDPR compliance.

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](/docs/CONTRIBUTING.md) for how to get involved. For setting up your development environment and the pull request process, refer to the guidelines.

## Community Guidelines

BabbleBeaver is committed to fostering an inclusive and safe community. Please read our [Community Guidelines](/docs/COMMUNITY_GUIDELINES.md) to understand our values and expectations.

## License

BabbleBeaver is GPL-licensed. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project was inspired by the open-source community and the need for privacy-respecting AI tools.

## Contact

For questions, suggestions, or contributions, please open an issue in this repository or contact us at [team@open.build](mailto:team@open.build).


## Why GPL for BabbleBeaver?
Copyleft: The GPL is a strong copyleft license, which means that any modified versions of the project must also be distributed under the GPL. This ensures that the main codebase and any derivatives remain open source, promoting collaboration and improvement.
Compatibility with Other Licenses for Sub-Repositories: While the GPL itself is strict about the licensing of derived works, it allows for linking or interfacing with software under different licenses under certain conditions.
