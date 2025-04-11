# PDF QA System

A powerful backend system for processing and answering questions about PDF documents using FastAPI and Anthropic's Claude AI.

## Features

- PDF document processing and text extraction
- AI-powered question answering about PDF content
- Secure authentication system
- RESTful API endpoints
- Docker containerization support

## Tech Stack

- **Backend Framework**: FastAPI
- **Language**: Python 3.10
- **AI Integration**: Anthropic Claude
- **Authentication**: JWT with python-jose
- **Database**: Supabase
- **PDF Processing**: PyPDF2
- **Containerization**: Docker

## Prerequisites

- Python 3.10
- Docker (optional)
- Rust (for building dependencies)

## Installation

### Local Development

1. Clone the repository:
```bash
git clone <repository-url>
cd pdf-qa-system/backend
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Docker Setup

1. Build the Docker image:
```bash
docker build -t pdf-qa-system .
```

2. Run the container:
```bash
docker run -p 8000:8000 pdf-qa-system
```

## Environment Variables

Create a `.env` file in the root directory with the following variables:

```
ANTHROPIC_API_KEY=your_api_key
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
JWT_SECRET_KEY=your_jwt_secret
```

## API Endpoints

- `POST /upload`: Upload PDF documents
- `POST /ask`: Ask questions about uploaded PDFs
- `POST /auth/login`: User authentication
- `POST /auth/register`: User registration

## Development

To run the development server:

```bash
uvicorn app:app --reload
```

The API will be available at `http://localhost:8000`

## API Documentation

Once the server is running, you can access the interactive API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 