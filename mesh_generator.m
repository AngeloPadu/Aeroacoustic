function create_rectangular_mesh_nas(center, width, height, depth, nx, ny, nz, filename)
    % center: [x, y] coordinate of the center of the rectangle in the xy-plane
    % width: width of the rectangle in the x-direction
    % height: height of the rectangle in the y-direction
    % depth: extrusion depth in the z-direction
    % nx: number of elements in the x-direction
    % ny: number of elements in the y-direction
    % nz: number of elements in the z-direction
    % filename: name of the .nas file to save

    % Calculate the coordinates of the corners of the rectangle
    x0 = center(1) - width / 2;
    y0 = center(2) - height / 2;
    x1 = center(1) + width / 2;
    y1 = center(2) + height / 2;
    

     x = linspace(x0, x1, nx);
    delta = (x(2)-x(1));
    ny = floor((y1-y0)/delta);
    % Create the meshgrid in the xy-plane
    [X, Y] = meshgrid(x, linspace(y0, y1, ny));
    Z0 = ones(size(X))*center(3);
    Z1 = Z0 + depth;
    
    % Create the vertices for the bottom and top faces
    vertices_bottom = [X(:), Y(:), Z0(:)];
    vertices_top = [X(:), Y(:), Z1(:)];
    
    % Combine bottom and top vertices
    vertices = [vertices_bottom; vertices_top];
    

    

    % Create the faces
    faces = [];
    
    % Bottom face indices
    for j = 1:ny-1
        for i = 1:nx-1
            % Bottom face
            faces = [faces; ...
                sub2ind([ny, nx], j, i), ...
                sub2ind([ny, nx], j, i+1), ...
                sub2ind([ny, nx], j+1, i+1), ...
                sub2ind([ny, nx], j+1, i)];
        end
    end
    
    % Top face indices
    offset = nx * ny;
    for j = 1:ny-1
        for i = 1:nx-1
            faces = [faces; ...
                offset + sub2ind([ny, nx], j, i), ...
                offset + sub2ind([ny, nx], j, i+1), ...
                offset + sub2ind([ny, nx], j+1, i+1), ...
                offset + sub2ind([ny, nx], j+1, i)];
        end
    end
    
    % Side faces indices
    for j = 1:ny-1
        for i = 1:nx-1
            % Sides faces
            faces = [faces; ...
                sub2ind([ny, nx], j, i), ...
                sub2ind([ny, nx], j+1, i), ...
                offset + sub2ind([ny, nx], j+1, i), ...
                offset + sub2ind([ny, nx], j, i)];

            faces = [faces; ...
                sub2ind([ny, nx], j, i), ...
                offset + sub2ind([ny, nx], j, i), ...
                offset + sub2ind([ny, nx], j, i+1), ...
                sub2ind([ny, nx], j, i+1)];

            faces = [faces; ...
                sub2ind([ny, nx], j, i+1), ...
                offset + sub2ind([ny, nx], j, i+1), ...
                offset + sub2ind([ny, nx], j+1, i+1), ...
                sub2ind([ny, nx], j+1, i+1)];

            faces = [faces; ...
                sub2ind([ny, nx], j+1, i), ...
                sub2ind([ny, nx], j+1, i+1), ...
                offset + sub2ind([ny, nx], j+1, i+1), ...
                offset + sub2ind([ny, nx], j+1, i)];
        end
    end
    

    % Plot the mesh with quad faces
    figure;
    hold on;
    for k = 1:size(faces, 1)
        face_vertices = vertices(faces(k, :));
        patch('Vertices', vertices, 'Faces', faces(k, :), 'FaceColor', 'cyan', 'FaceAlpha', 0.8, 'EdgeColor', 'black');
    end
    xlabel('X');
    ylabel('Y');
    zlabel('Z');
    title('Rectangular Mesh with Quad Faces');
    axis equal;
    grid on;
    %view(3);
    hold off;
    
    % Write to Nastran file
    write_nas(vertices, faces, filename);
end

function write_nas(vertices, faces, filename)
    fid = fopen(filename, 'w');
    
    % Write header
    fprintf(fid, 'CEND\n');
    fprintf(fid, 'BEGIN BULK\n');
    
    % Write nodes
    num_points = size(vertices, 1);
    for i = 1:num_points
        fprintf(fid, 'GRID,%d,%d,%.6f,%.6f,%.6f\n', i, 0, vertices(i, 1), vertices(i, 2), vertices(i, 3));
    end
    
    % Write elements (quad faces)
    num_faces = size(faces, 1);
    for i = 1:num_faces
        fprintf(fid, 'CQUAD4,%d,%d,%d,%d,%d,%d\n', i, 1, faces(i, 1), faces(i, 2), faces(i, 3), faces(i, 4));
    end
    
    % Write end of file
    fprintf(fid, 'ENDDATA\n');
    
    fclose(fid);
end

% Example usage
center = [0.006327922, 0,0.006223];
width = 9.7e-03;
height = 0.04;
depth = 0.001;
nx = 200;
ny = 5;
nz = 1; % Not used in this code, kept for generality
filename = 'rectangular_mesh.nas';
create_rectangular_mesh_nas(center, width, height, depth, nx, ny, nz, filename);
