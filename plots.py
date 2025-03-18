# -*- coding: utf-8 -*-
"""
Created on Sun Nov  3 15:27:27 2024

@author: Abdelrahman
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_decision_regions(equal_priors=True):
    # Create a grid of points
    x1 = np.linspace(-4, 4, 400)
    x2 = np.linspace(-4, 4, 400)
    X1, X2 = np.meshgrid(x1, x2)
    
    # Calculate radius for decision boundary
    if equal_priors:
        r = np.sqrt((8/3) * np.log(2))
        title = "Decision Regions (Equal Priors)"
    else:
        r = np.sqrt((8/3) * np.log(8/3))
        title = "Decision Regions (Unequal Priors P(w₁)=1/4, P(w₂)=3/4)"
    
    # Calculate decision regions
    Z = X1**2 + X2**2
    region = Z <= r**2
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Plot decision regions
    plt.contourf(X1, X2, region, alpha=0.3, colors=['lightblue', 'lightgreen'])
    
    # Plot decision boundary
    theta = np.linspace(0, 2*np.pi, 100)
    circle_x = r * np.cos(theta)
    circle_y = r * np.sin(theta)
    plt.plot(circle_x, circle_y, 'k--', label=f'Decision Boundary (r ≈ {r:.2f})')
    
    # Plot equal probability contours for each class
    # For class 1 (Σ₁ = I)
    for k in [1, 2, 3]:
        circle = plt.Circle((0, 0), k, fill=False, color='blue', linestyle=':')
        plt.gca().add_artist(circle)
    
    # For class 2 (Σ₂ = 4I)
    for k in [2, 4, 6]:
        circle = plt.Circle((0, 0), k, fill=False, color='green', linestyle=':')
        plt.gca().add_artist(circle)
    
    # Add labels and title
    plt.xlabel('x₁')
    plt.ylabel('x₂')
    plt.title(title)
    plt.axis('equal')
    plt.grid(True)
    
    # Add legend with custom patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightblue', alpha=0.3, label='Class w₁ Region'),
        Patch(facecolor='lightgreen', alpha=0.3, label='Class w₂ Region'),
        plt.Line2D([0], [0], color='k', linestyle='--', label='Decision Boundary'),
        plt.Line2D([0], [0], color='blue', linestyle=':', label='Class w₁ Contours'),
        plt.Line2D([0], [0], color='green', linestyle=':', label='Class w₂ Contours')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.axis([-4, 4, -4, 4])
    plt.show()

# Plot both cases
plt.style.use('seaborn')

# Case 1: Equal Priors
plot_decision_regions(equal_priors=True)

# Case 2: Unequal Priors
plot_decision_regions(equal_priors=False)

# Add additional visualization showing both boundaries on same plot
plt.figure(figsize=(10, 8))

# Calculate both radii
r1 = np.sqrt((8/3) * np.log(2))
r2 = np.sqrt((8/3) * np.log(8/3))

# Plot both decision boundaries
theta = np.linspace(0, 2*np.pi, 100)
plt.plot(r1 * np.cos(theta), r1 * np.sin(theta), 'b--', 
         label=f'Equal Priors (r ≈ {r1:.2f})')
plt.plot(r2 * np.cos(theta), r2 * np.sin(theta), 'r--', 
         label=f'Unequal Priors (r ≈ {r2:.2f})')

plt.xlabel('x₁')
plt.ylabel('x₂')
plt.title('Comparison of Decision Boundaries')
plt.axis('equal')
plt.grid(True)
plt.legend()
plt.axis([-4, 4, -4, 4])
plt.show()