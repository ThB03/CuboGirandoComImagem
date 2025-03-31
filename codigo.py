import cv2
import numpy as np

# Define the cube's vertices in 3D space
cube_vertices = np.array([
    [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],  # Back face
    [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1]   # Front face
], dtype=np.float32)

# Define the cube's faces using vertex indices
cube_faces = [
    [0, 1, 2, 3],  # Back
    [4, 5, 6, 7],  # Front
    [0, 1, 5, 4],  # Bottom
    [3, 2, 6, 7],  # Top
    [0, 3, 7, 4],  # Left
    [1, 2, 6, 5]   # Right
]

# Define texture corners for mapping
texture_corners = np.array([[0, 0], [200, 0], [200, 200], [0, 200]], dtype=np.float32)

def get_rotation_matrix(angle_x, angle_y, angle_z):
    """Generates a 3D rotation matrix from angles in degrees."""
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(angle_x), -np.sin(angle_x)],
        [0, np.sin(angle_x), np.cos(angle_x)]
    ])
    
    Ry = np.array([
        [np.cos(angle_y), 0, np.sin(angle_y)],
        [0, 1, 0],
        [-np.sin(angle_y), 0, np.cos(angle_y)]
    ])
    
    Rz = np.array([
        [np.cos(angle_z), -np.sin(angle_z), 0],
        [np.sin(angle_z), np.cos(angle_z), 0],
        [0, 0, 1]
    ])
    
    return Rz @ Ry @ Rx  # Combined rotation

def project_3d_to_2d(points_3d, img_size=500):
    """Projects 3D points onto a 2D plane using perspective projection."""
    projected_points = []
    for p in points_3d:
        x, y, z = p
        z += 5  # Avoid division by zero
        x_2d = int(img_size / 2 + (x / z) * img_size)
        y_2d = int(img_size / 2 - (y / z) * img_size)
        projected_points.append([x_2d, y_2d])
    
    return np.array(projected_points, dtype=np.float32)

def warp_with_homography(src_img, homography_matrix, output_size):
    """Applies a homography using vectorized operations."""
    height, width = output_size
    is_color = len(src_img.shape) == 3  # Check if image is color (multi-channel)

    # Create empty output image
    if is_color:
        dst = np.zeros((height, width, src_img.shape[2]), dtype=src_img.dtype)
    else:
        dst = np.zeros((height, width), dtype=src_img.dtype)

    # Create a grid of destination pixel coordinates
    oy, ox = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    
    dest_coords = np.vstack((ox.ravel(), oy.ravel(), np.ones(ox.size)))
    
    # Apply the homography transformation to all pixels
    xw, yw, w = homography_matrix.dot(dest_coords)
    
    epsilon = 10**-4
    w = np.where(np.abs(w) < epsilon, epsilon, w)  # Set small w values to epsilon

    tx = (xw / w).astype(int)
    ty = (yw / w).astype(int)
    
    # Create a mask for valid coordinates
    valid_mask = (tx >= 0) & (tx < src_img.shape[1]) & (ty >= 0) & (ty < src_img.shape[0])
    
    # Get the valid indices
    valid_indices = np.where(valid_mask.ravel())[0]

    # Map the valid source pixels to destination pixels
    if is_color:
        dst[oy.ravel()[valid_indices], ox.ravel()[valid_indices], :] = src_img[ty[valid_indices], tx[valid_indices], :]
    else:
        dst[oy.ravel()[valid_indices], ox.ravel()[valid_indices]] = src_img[ty[valid_indices], tx[valid_indices]]

    return dst

def render_cube(textures, angle_x, angle_y, angle_z):
    """Renders a spinning cube with texture mapping."""
    # Rotate the cube
    R = get_rotation_matrix(np.radians(angle_x), np.radians(angle_y), np.radians(angle_z))
    rotated_vertices = np.dot(cube_vertices, R.T)
    
    face_depths = []
    for face in cube_faces:
        avg_depth = np.mean(rotated_vertices[face, 2])  # Average Z-depth of face
        face_depths.append((avg_depth, face))

    face_depths.sort(reverse=True, key=lambda x: x[0])
    
    # Project 3D to 2D
    projected_vertices = project_3d_to_2d(rotated_vertices)

    img = np.zeros((500, 500, 3), dtype=np.uint8)

    # Draw and texture each face
    for _, face in face_depths:
        face_2d = projected_vertices[face]

        H, _ = cv2.findHomography(texture_corners, face_2d)

        # Get the corresponding texture for this face
        face_index = cube_faces.index(face)
        texture = textures[face_index]

        # Apply the homography
        warped_texture = warp_with_homography(texture, np.linalg.inv(H), (500, 500))
        
        # Create mask
        mask = (warped_texture != 0).any(axis=2).astype(np.uint8) * 255

        mask_3d = np.stack([mask] * img.shape[2], axis=2)

        # Apply the texture according to mask
        img = np.where(mask_3d > 0, warped_texture, img)

    return img

# Load textures
textures = [
    cv2.imread("th.jpg"),
    cv2.imread("bulcao.jpg"),
    cv2.imread("cosenza.jpg"),
    cv2.imread("hamilton.jpg"),
    cv2.imread("laxe.jpg"),
    cv2.imread("vieira.jpg")
]

# Resize textures to fit the faces of the cube
textures = [cv2.resize(texture, (200, 200)) for texture in textures]

# Animation loop
angle_x, angle_y, angle_z = 0, 0, 0
while True:
    angle_x += 2
    angle_y += 3
    angle_z += 1

    frame = render_cube(textures, angle_x, angle_y, angle_z)

    if np.any(frame[0, :] != 0) or np.any(frame[-1, :] != 0) or np.any(frame[:, 0] != 0) or np.any(frame[:, -1] != 0):
        continue
    
    # Show the animation
    cv2.imshow("Spinning Cube", frame)
    if cv2.waitKey(30) & 0xFF == 27:  # Press ESC to exit
        break

cv2.destroyAllWindows()
