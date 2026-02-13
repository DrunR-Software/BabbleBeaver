"""
D-Hub API Module

Community feed endpoints for the DrunR app.
Uses BigQuery for data storage with optional mock fallback.

Endpoints:
- GET  /dhub/posts           - Fetch paginated posts
- GET  /dhub/posts/{id}/replies - Fetch replies for a post
- POST /dhub/posts           - Create a new post
- POST /dhub/posts/{id}/replies - Create a reply
- DELETE /dhub/posts/{id}    - Delete a post
- POST /dhub/posts/{id}/like - Toggle like on a post
"""

import os
import logging
import uuid
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, Field

from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from google.cloud import bigquery

logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

# Set to False to use BigQuery, True to use mock data
USE_MOCK = os.getenv("DHUB_USE_MOCK", "true").lower() == "true"

# BigQuery dataset
DHUB_DATASET = os.getenv("DHUB_DATASET", "dhub")
PROJECT_ID = os.getenv("PROJECT_ID")

# Page size for pagination
PAGE_SIZE = 10

# ============================================================================
# Pydantic Models
# ============================================================================

class UserModel(BaseModel):
    id: str
    name: str
    avatar_url: Optional[str] = None
    verified: bool = False
    badges: List[str] = []


class FoodCardModel(BaseModel):
    id: str
    name: str
    image_url: Optional[str] = None
    carbs: int = 0
    fat: int = 0
    protein: int = 0
    calories: int = 0


class PostContentModel(BaseModel):
    text: str
    food_card: Optional[FoodCardModel] = None


class PostModel(BaseModel):
    id: str
    user: UserModel
    text: str
    food_card: Optional[FoodCardModel] = None
    created_at: str
    reply_count: int = 0
    like_count: int = 0
    is_liked: bool = False


class ReplyModel(BaseModel):
    id: str
    post_id: str
    user: UserModel
    text: str
    created_at: str


class PaginatedPostsResponse(BaseModel):
    count: int
    next: Optional[int] = None
    previous: Optional[int] = None
    results: List[PostModel]


class CreatePostRequest(BaseModel):
    text: str
    food_card: Optional[FoodCardModel] = None


class CreateReplyRequest(BaseModel):
    text: str


# ============================================================================
# Mock Data (for testing without BigQuery)
# ============================================================================

MOCK_USERS = {
    "user-001": UserModel(
        id="user-001",
        name="Alice Chen",
        avatar_url="https://i.pravatar.cc/100?u=alice",
        verified=False,
        badges=[]
    ),
    "user-002": UserModel(
        id="user-002",
        name="Dr. Smith",
        avatar_url="https://i.pravatar.cc/100?u=drsmith",
        verified=True,
        badges=["Nutritionist"]
    ),
    "user-003": UserModel(
        id="user-003",
        name="Chef Maria",
        avatar_url="https://i.pravatar.cc/100?u=maria",
        verified=True,
        badges=["Chef"]
    ),
}

MOCK_FOOD_CARDS = {
    "food-001": FoodCardModel(
        id="food-001",
        name="Garden Salad",
        image_url="https://images.unsplash.com/photo-1512621776951-a57141f2eefd?w=200",
        carbs=70,
        fat=20,
        protein=10,
        calories=180
    ),
    "food-002": FoodCardModel(
        id="food-002",
        name="Grilled Chicken Bowl",
        image_url="https://images.unsplash.com/photo-1546069901-ba9599a7e63c?w=200",
        carbs=45,
        fat=12,
        protein=42,
        calories=520
    ),
}

# In-memory mock storage
_mock_posts: List[dict] = []
_mock_replies: dict = {}
_mock_likes: dict = {}


def _init_mock_data():
    """Initialize mock data if empty."""
    global _mock_posts, _mock_replies

    if _mock_posts:
        return

    now = datetime.utcnow()

    _mock_posts = [
        {
            "id": "post-001",
            "user_id": "user-001",
            "text": "Just tried this amazing salad for lunch! The dressing is to die for.",
            "food_card_id": "food-001",
            "reply_count": 2,
            "like_count": 15,
            "created_at": now.isoformat(),
        },
        {
            "id": "post-002",
            "user_id": "user-002",
            "text": "Meal prep Sunday! Here's my go-to protein bowl that's both delicious and nutritious.",
            "food_card_id": "food-002",
            "reply_count": 3,
            "like_count": 42,
            "created_at": now.isoformat(),
        },
        {
            "id": "post-003",
            "user_id": "user-003",
            "text": "Quick tip: Adding fresh herbs can transform any dish. What are your favorites?",
            "food_card_id": None,
            "reply_count": 5,
            "like_count": 89,
            "created_at": now.isoformat(),
        },
    ]

    _mock_replies = {
        "post-001": [
            {
                "id": "reply-001",
                "post_id": "post-001",
                "user_id": "user-002",
                "text": "Great macro balance! The leafy greens provide excellent micronutrients.",
                "created_at": now.isoformat(),
            },
            {
                "id": "reply-002",
                "post_id": "post-001",
                "user_id": "user-003",
                "text": "This looks so fresh! Where did you get the ingredients?",
                "created_at": now.isoformat(),
            },
        ],
        "post-002": [
            {
                "id": "reply-003",
                "post_id": "post-002",
                "user_id": "user-001",
                "text": "Perfect for carb loading before a workout!",
                "created_at": now.isoformat(),
            },
        ],
    }


def _mock_post_to_model(post: dict, current_user_id: str) -> PostModel:
    """Convert mock post dict to PostModel."""
    user = MOCK_USERS.get(post["user_id"], MOCK_USERS["user-001"])
    food_card = MOCK_FOOD_CARDS.get(post.get("food_card_id")) if post.get("food_card_id") else None
    is_liked = current_user_id in _mock_likes.get(post["id"], set())

    return PostModel(
        id=post["id"],
        user=user,
        text=post["text"],
        food_card=food_card,
        created_at=post["created_at"],
        reply_count=post["reply_count"],
        like_count=post["like_count"],
        is_liked=is_liked
    )


def _mock_reply_to_model(reply: dict) -> ReplyModel:
    """Convert mock reply dict to ReplyModel."""
    user = MOCK_USERS.get(reply["user_id"], MOCK_USERS["user-001"])

    return ReplyModel(
        id=reply["id"],
        post_id=reply["post_id"],
        user=user,
        text=reply["text"],
        created_at=reply["created_at"]
    )


# ============================================================================
# BigQuery Operations
# ============================================================================

def get_bq_client():
    """Get BigQuery client."""
    return bigquery.Client(project=PROJECT_ID)


def bq_fetch_posts(page: int, current_user_id: str) -> PaginatedPostsResponse:
    """Fetch paginated posts from BigQuery."""
    client = get_bq_client()
    offset = (page - 1) * PAGE_SIZE

    # Get total count
    count_query = f"""
        SELECT COUNT(*) as total
        FROM `{PROJECT_ID}.{DHUB_DATASET}.posts`
        WHERE is_deleted = FALSE
    """
    count_result = client.query(count_query).result()
    total_count = list(count_result)[0].total

    # Get posts with user and food card info
    query = f"""
        SELECT
            p.id,
            p.text,
            p.reply_count,
            p.like_count,
            p.created_at,
            STRUCT(u.id, u.name, u.avatar_url, u.verified, u.badges) AS user,
            CASE
                WHEN fc.id IS NOT NULL THEN STRUCT(
                    fc.id, fc.name, fc.image_url, fc.carbs, fc.fat, fc.protein, fc.calories
                )
                ELSE NULL
            END AS food_card,
            EXISTS(
                SELECT 1 FROM `{PROJECT_ID}.{DHUB_DATASET}.post_likes` pl
                WHERE pl.post_id = p.id AND pl.user_id = @current_user_id
            ) AS is_liked
        FROM `{PROJECT_ID}.{DHUB_DATASET}.posts` p
        JOIN `{PROJECT_ID}.{DHUB_DATASET}.users` u ON p.user_id = u.id
        LEFT JOIN `{PROJECT_ID}.{DHUB_DATASET}.food_cards` fc ON p.food_card_id = fc.id
        WHERE p.is_deleted = FALSE
        ORDER BY p.created_at DESC
        LIMIT @page_size OFFSET @offset
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("current_user_id", "STRING", current_user_id),
            bigquery.ScalarQueryParameter("page_size", "INT64", PAGE_SIZE),
            bigquery.ScalarQueryParameter("offset", "INT64", offset),
        ]
    )

    results = client.query(query, job_config=job_config).result()

    posts = []
    for row in results:
        posts.append(PostModel(
            id=row.id,
            user=UserModel(**dict(row.user)),
            text=row.text,
            food_card=FoodCardModel(**dict(row.food_card)) if row.food_card else None,
            created_at=row.created_at.isoformat(),
            reply_count=row.reply_count,
            like_count=row.like_count,
            is_liked=row.is_liked
        ))

    return PaginatedPostsResponse(
        count=total_count,
        next=page + 1 if offset + PAGE_SIZE < total_count else None,
        previous=page - 1 if page > 1 else None,
        results=posts
    )


def bq_fetch_replies(post_id: str) -> List[ReplyModel]:
    """Fetch replies for a post from BigQuery."""
    client = get_bq_client()

    query = f"""
        SELECT
            r.id,
            r.post_id,
            r.text,
            r.created_at,
            STRUCT(u.id, u.name, u.avatar_url, u.verified, u.badges) AS user
        FROM `{PROJECT_ID}.{DHUB_DATASET}.replies` r
        JOIN `{PROJECT_ID}.{DHUB_DATASET}.users` u ON r.user_id = u.id
        WHERE r.post_id = @post_id AND r.is_deleted = FALSE
        ORDER BY r.created_at ASC
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("post_id", "STRING", post_id),
        ]
    )

    results = client.query(query, job_config=job_config).result()

    replies = []
    for row in results:
        replies.append(ReplyModel(
            id=row.id,
            post_id=row.post_id,
            user=UserModel(**dict(row.user)),
            text=row.text,
            created_at=row.created_at.isoformat()
        ))

    return replies


def bq_create_post(user_id: str, text: str, food_card: Optional[FoodCardModel]) -> PostModel:
    """Create a new post in BigQuery."""
    client = get_bq_client()
    post_id = str(uuid.uuid4())
    now = datetime.utcnow()

    # If food_card provided, insert it first
    food_card_id = None
    if food_card:
        food_card_id = food_card.id or str(uuid.uuid4())
        fc_query = f"""
            INSERT INTO `{PROJECT_ID}.{DHUB_DATASET}.food_cards`
            (id, name, image_url, carbs, fat, protein, calories)
            VALUES (@id, @name, @image_url, @carbs, @fat, @protein, @calories)
        """
        fc_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("id", "STRING", food_card_id),
                bigquery.ScalarQueryParameter("name", "STRING", food_card.name),
                bigquery.ScalarQueryParameter("image_url", "STRING", food_card.image_url),
                bigquery.ScalarQueryParameter("carbs", "INT64", food_card.carbs),
                bigquery.ScalarQueryParameter("fat", "INT64", food_card.fat),
                bigquery.ScalarQueryParameter("protein", "INT64", food_card.protein),
                bigquery.ScalarQueryParameter("calories", "INT64", food_card.calories),
            ]
        )
        client.query(fc_query, job_config=fc_config).result()

    # Insert post
    query = f"""
        INSERT INTO `{PROJECT_ID}.{DHUB_DATASET}.posts`
        (id, user_id, text, food_card_id, reply_count, like_count, is_deleted, created_at, updated_at)
        VALUES (@id, @user_id, @text, @food_card_id, 0, 0, FALSE, @created_at, @created_at)
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("id", "STRING", post_id),
            bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
            bigquery.ScalarQueryParameter("text", "STRING", text),
            bigquery.ScalarQueryParameter("food_card_id", "STRING", food_card_id),
            bigquery.ScalarQueryParameter("created_at", "TIMESTAMP", now),
        ]
    )

    client.query(query, job_config=job_config).result()

    # Fetch user info for response
    user_query = f"""
        SELECT id, name, avatar_url, verified, badges
        FROM `{PROJECT_ID}.{DHUB_DATASET}.users`
        WHERE id = @user_id
    """
    user_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
        ]
    )
    user_result = list(client.query(user_query, job_config=user_config).result())

    if user_result:
        user = UserModel(**dict(user_result[0]))
    else:
        user = UserModel(id=user_id, name="Unknown User")

    return PostModel(
        id=post_id,
        user=user,
        text=text,
        food_card=food_card,
        created_at=now.isoformat(),
        reply_count=0,
        like_count=0,
        is_liked=False
    )


def bq_create_reply(post_id: str, user_id: str, text: str) -> ReplyModel:
    """Create a reply in BigQuery."""
    client = get_bq_client()
    reply_id = str(uuid.uuid4())
    now = datetime.utcnow()

    # Insert reply
    query = f"""
        INSERT INTO `{PROJECT_ID}.{DHUB_DATASET}.replies`
        (id, post_id, user_id, text, is_deleted, created_at)
        VALUES (@id, @post_id, @user_id, @text, FALSE, @created_at)
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("id", "STRING", reply_id),
            bigquery.ScalarQueryParameter("post_id", "STRING", post_id),
            bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
            bigquery.ScalarQueryParameter("text", "STRING", text),
            bigquery.ScalarQueryParameter("created_at", "TIMESTAMP", now),
        ]
    )

    client.query(query, job_config=job_config).result()

    # Update reply count on post
    update_query = f"""
        UPDATE `{PROJECT_ID}.{DHUB_DATASET}.posts`
        SET reply_count = reply_count + 1, updated_at = CURRENT_TIMESTAMP()
        WHERE id = @post_id
    """
    update_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("post_id", "STRING", post_id),
        ]
    )
    client.query(update_query, job_config=update_config).result()

    # Fetch user info
    user_query = f"""
        SELECT id, name, avatar_url, verified, badges
        FROM `{PROJECT_ID}.{DHUB_DATASET}.users`
        WHERE id = @user_id
    """
    user_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
        ]
    )
    user_result = list(client.query(user_query, job_config=user_config).result())

    if user_result:
        user = UserModel(**dict(user_result[0]))
    else:
        user = UserModel(id=user_id, name="Unknown User")

    return ReplyModel(
        id=reply_id,
        post_id=post_id,
        user=user,
        text=text,
        created_at=now.isoformat()
    )


def bq_delete_post(post_id: str, user_id: str) -> bool:
    """Soft delete a post in BigQuery."""
    client = get_bq_client()

    query = f"""
        UPDATE `{PROJECT_ID}.{DHUB_DATASET}.posts`
        SET is_deleted = TRUE, updated_at = CURRENT_TIMESTAMP()
        WHERE id = @post_id AND user_id = @user_id
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("post_id", "STRING", post_id),
            bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
        ]
    )

    result = client.query(query, job_config=job_config).result()
    return True


def bq_toggle_like(post_id: str, user_id: str) -> PostModel:
    """Toggle like on a post in BigQuery."""
    client = get_bq_client()

    # Check if already liked
    check_query = f"""
        SELECT 1 FROM `{PROJECT_ID}.{DHUB_DATASET}.post_likes`
        WHERE post_id = @post_id AND user_id = @user_id
    """
    check_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("post_id", "STRING", post_id),
            bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
        ]
    )
    check_result = list(client.query(check_query, job_config=check_config).result())

    if check_result:
        # Unlike - remove the like
        delete_query = f"""
            DELETE FROM `{PROJECT_ID}.{DHUB_DATASET}.post_likes`
            WHERE post_id = @post_id AND user_id = @user_id
        """
        client.query(delete_query, job_config=check_config).result()
    else:
        # Like - add the like
        insert_query = f"""
            INSERT INTO `{PROJECT_ID}.{DHUB_DATASET}.post_likes`
            (post_id, user_id, created_at)
            VALUES (@post_id, @user_id, CURRENT_TIMESTAMP())
        """
        client.query(insert_query, job_config=check_config).result()

    # Update cached like_count
    update_query = f"""
        UPDATE `{PROJECT_ID}.{DHUB_DATASET}.posts`
        SET like_count = (
            SELECT COUNT(*) FROM `{PROJECT_ID}.{DHUB_DATASET}.post_likes`
            WHERE post_id = @post_id
        )
        WHERE id = @post_id
    """
    client.query(update_query, job_config=check_config).result()

    # Fetch updated post
    posts = bq_fetch_posts(1, user_id)
    for post in posts.results:
        if post.id == post_id:
            return post

    raise HTTPException(status_code=404, detail="Post not found")


# ============================================================================
# FastAPI Router
# ============================================================================

router = APIRouter(prefix="/dhub", tags=["D-Hub"])


def get_current_user_id(request: Request) -> str:
    """
    Extract current user ID from request.
    In production, this should come from JWT token.
    """
    # Try to get from header (set by auth middleware)
    user_id = request.headers.get("X-User-ID")
    if user_id:
        return user_id

    # Try to get from query param (for testing)
    user_id = request.query_params.get("user_id")
    if user_id:
        return user_id

    # Default for testing
    return "user-001"


@router.get("/posts", response_model=PaginatedPostsResponse)
async def get_posts(request: Request, page: int = 1):
    """
    Fetch paginated posts.

    Query Parameters:
        page: Page number (default: 1)
        user_id: Current user ID (for testing)
    """
    current_user_id = get_current_user_id(request)

    if USE_MOCK:
        _init_mock_data()

        offset = (page - 1) * PAGE_SIZE
        paginated = _mock_posts[offset:offset + PAGE_SIZE]

        return PaginatedPostsResponse(
            count=len(_mock_posts),
            next=page + 1 if offset + PAGE_SIZE < len(_mock_posts) else None,
            previous=page - 1 if page > 1 else None,
            results=[_mock_post_to_model(p, current_user_id) for p in paginated]
        )

    try:
        return bq_fetch_posts(page, current_user_id)
    except Exception as e:
        logger.error(f"Error fetching posts: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch posts")


@router.get("/posts/{post_id}/replies", response_model=List[ReplyModel])
async def get_replies(post_id: str):
    """Fetch replies for a specific post."""
    if USE_MOCK:
        _init_mock_data()

        replies = _mock_replies.get(post_id, [])
        return [_mock_reply_to_model(r) for r in replies]

    try:
        return bq_fetch_replies(post_id)
    except Exception as e:
        logger.error(f"Error fetching replies: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch replies")


@router.post("/posts", response_model=PostModel)
async def create_post(request: Request, body: CreatePostRequest):
    """Create a new post."""
    current_user_id = get_current_user_id(request)

    if USE_MOCK:
        _init_mock_data()

        new_post = {
            "id": f"post-{uuid.uuid4().hex[:8]}",
            "user_id": current_user_id,
            "text": body.text,
            "food_card_id": body.food_card.id if body.food_card else None,
            "reply_count": 0,
            "like_count": 0,
            "created_at": datetime.utcnow().isoformat(),
        }

        # Add food card to mock storage if provided
        if body.food_card:
            MOCK_FOOD_CARDS[body.food_card.id] = body.food_card

        _mock_posts.insert(0, new_post)
        _mock_replies[new_post["id"]] = []

        return _mock_post_to_model(new_post, current_user_id)

    try:
        return bq_create_post(current_user_id, body.text, body.food_card)
    except Exception as e:
        logger.error(f"Error creating post: {e}")
        raise HTTPException(status_code=500, detail="Failed to create post")


@router.post("/posts/{post_id}/replies", response_model=ReplyModel)
async def create_reply(request: Request, post_id: str, body: CreateReplyRequest):
    """Create a reply to a post."""
    current_user_id = get_current_user_id(request)

    if USE_MOCK:
        _init_mock_data()

        # Check post exists
        post = next((p for p in _mock_posts if p["id"] == post_id), None)
        if not post:
            raise HTTPException(status_code=404, detail="Post not found")

        new_reply = {
            "id": f"reply-{uuid.uuid4().hex[:8]}",
            "post_id": post_id,
            "user_id": current_user_id,
            "text": body.text,
            "created_at": datetime.utcnow().isoformat(),
        }

        if post_id not in _mock_replies:
            _mock_replies[post_id] = []
        _mock_replies[post_id].append(new_reply)
        post["reply_count"] += 1

        return _mock_reply_to_model(new_reply)

    try:
        return bq_create_reply(post_id, current_user_id, body.text)
    except Exception as e:
        logger.error(f"Error creating reply: {e}")
        raise HTTPException(status_code=500, detail="Failed to create reply")


@router.delete("/posts/{post_id}")
async def delete_post(request: Request, post_id: str):
    """Delete a post (soft delete)."""
    current_user_id = get_current_user_id(request)

    if USE_MOCK:
        _init_mock_data()

        post_idx = next(
            (i for i, p in enumerate(_mock_posts)
             if p["id"] == post_id and p["user_id"] == current_user_id),
            None
        )

        if post_idx is None:
            raise HTTPException(status_code=404, detail="Post not found or not authorized")

        _mock_posts.pop(post_idx)
        _mock_replies.pop(post_id, None)

        return {"success": True}

    try:
        bq_delete_post(post_id, current_user_id)
        return {"success": True}
    except Exception as e:
        logger.error(f"Error deleting post: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete post")


@router.post("/posts/{post_id}/like", response_model=PostModel)
async def toggle_like(request: Request, post_id: str):
    """Toggle like on a post."""
    current_user_id = get_current_user_id(request)

    if USE_MOCK:
        _init_mock_data()

        post = next((p for p in _mock_posts if p["id"] == post_id), None)
        if not post:
            raise HTTPException(status_code=404, detail="Post not found")

        if post_id not in _mock_likes:
            _mock_likes[post_id] = set()

        if current_user_id in _mock_likes[post_id]:
            _mock_likes[post_id].remove(current_user_id)
            post["like_count"] -= 1
        else:
            _mock_likes[post_id].add(current_user_id)
            post["like_count"] += 1

        return _mock_post_to_model(post, current_user_id)

    try:
        return bq_toggle_like(post_id, current_user_id)
    except Exception as e:
        logger.error(f"Error toggling like: {e}")
        raise HTTPException(status_code=500, detail="Failed to toggle like")


# ============================================================================
# Health check endpoint
# ============================================================================

@router.get("/health")
async def health_check():
    """Health check for D-Hub service."""
    return {
        "status": "healthy",
        "service": "dhub",
        "mock_mode": USE_MOCK,
        "timestamp": datetime.utcnow().isoformat()
    }
